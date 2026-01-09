import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import KFold
import pyro

def tune_vae_optuna(data, model_search_space, n_splits=3, gap_windows=5, 
                    n_trials=20, timeout=None):

    kf = KFold(n_splits=n_splits, shuffle=False)
    device = data["device"]
    
    # Preparar dados
    if data["mode"] == "tensor":
        X_all = data["X_train"] 
        total_samples = X_all.size(0)
        print(f"Usando slicing direto em VRAM: {total_samples} samples")
    else:
        if "train_dataset" not in data:
            raise ValueError("Modo 'dataloader' requer 'train_dataset' para CV.")
        dataset = data["train_dataset"]
        total_samples = len(dataset)
        print("Usando DataLoader Subsets (mais lento)")

    indices = np.arange(total_samples)
    
    # Função objetivo para Optuna
    def objective(trial):
        # Sugerir hiperparâmetros baseado no model_search_space
        params = {}
        
        # z_dim
        if len(model_search_space['z_dim']) == 2:
            params['z_dim'] = trial.suggest_int('z_dim', 
                                                model_search_space['z_dim'][0],
                                                model_search_space['z_dim'][1])
        else:
            params['z_dim'] = model_search_space['z_dim'][0]
        
        # hidden_dim
        if len(model_search_space['hidden_dim']) == 2:
            params['hidden_dim'] = trial.suggest_int('hidden_dim',
                                                     model_search_space['hidden_dim'][0],
                                                     model_search_space['hidden_dim'][1])
        else:
            params['hidden_dim'] = model_search_space['hidden_dim'][0]
        
        # lr
        if len(model_search_space['lr']) == 2:
            params['lr'] = trial.suggest_float('lr',
                                               model_search_space['lr'][0],
                                               model_search_space['lr'][1],
                                               log=True)
        else:
            params['lr'] = model_search_space['lr'][0]
        
        # batch_size
        if len(model_search_space['batch_size']) > 1:
            params['batch_size'] = trial.suggest_categorical('batch_size',
                                                             model_search_space['batch_size'])
        else:
            params['batch_size'] = model_search_space['batch_size'][0]
        
        # epochs
        if len(model_search_space['epochs']) == 2:
            params['epochs'] = trial.suggest_int('epochs',
                                                 model_search_space['epochs'][0],
                                                 model_search_space['epochs'][1])
        else:
            params['epochs'] = model_search_space['epochs'][0]
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number + 1}: z_dim={params['z_dim']}, "
              f"hidden_dim={params['hidden_dim']}, lr={params['lr']:.2e}, "
              f"batch_size={params['batch_size']}, epochs={params['epochs']}")
        print(f"{'='*60}")
        
        fold_losses = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            pyro.clear_param_store()
            
            # Aplicar gap temporal
            val_start = val_idx.min()
            val_end = val_idx.max()
            mask_safe = ~((train_idx >= val_start - gap_windows) & 
                         (train_idx <= val_end + gap_windows))
            train_idx_purged = train_idx[mask_safe]
            
            # Criar modelo VAE
            vae = LogAnomalyVAE(
                input_dim=data['input_dim'],
                z_dim=params['z_dim'],
                hidden_dim=params['hidden_dim'],
                use_cuda=(device.type == "cuda")
            )
            
            optimizer = Adam({"lr": params['lr']})
            svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
            
            # Treinar e validar
            if data["mode"] == "tensor":
                X_fold_train = X_all[train_idx_purged]
                X_fold_val = X_all[val_idx]
                
                # Treinamento
                for epoch in range(params['epochs']):
                    epoch_loss = 0.0
                    num_samples = 0
                    
                    for x_batch in get_vram_batches(X_fold_train, 
                                                    params['batch_size'], 
                                                    shuffle=True):
                        loss = svi.step(x_batch)
                        epoch_loss += loss
                        num_samples += x_batch.size(0)
                    
                    if epoch == params['epochs'] - 1:
                        avg_epoch_loss = epoch_loss / num_samples
                        print(f"  Fold {fold_idx+1} - Época final: {avg_epoch_loss:.2f}")
                
                # Validação
                val_loss = 0.0
                val_samples = 0
                for x_val_batch in get_vram_batches(X_fold_val, 
                                                    batch_size=4096, 
                                                    shuffle=False):
                    val_loss += svi.evaluate_loss(x_val_batch)
                    val_samples += x_val_batch.size(0)
                
                avg_val_loss = val_loss / val_samples
                
            else:  # modo dataloader
                train_sub = Subset(dataset, train_idx_purged)
                val_sub = Subset(dataset, val_idx)
                
                train_loader = DataLoader(train_sub, 
                                        batch_size=params['batch_size'], 
                                        shuffle=True, 
                                        num_workers=0)
                val_loader = DataLoader(val_sub, 
                                      batch_size=4096, 
                                      shuffle=False, 
                                      num_workers=0)
                
                # Treinamento
                for epoch in range(params['epochs']):
                    epoch_loss = 0.0
                    num_samples = 0
                    for x_batch, *_ in train_loader:
                        x_batch = x_batch.to(device)
                        loss = svi.step(x_batch)
                        epoch_loss += loss
                        num_samples += x_batch.size(0)
                    
                    if epoch == params['epochs'] - 1:
                        avg_epoch_loss = epoch_loss / num_samples
                        print(f"  Fold {fold_idx+1} - Época final: {avg_epoch_loss:.2f}")
                
                # Validação
                val_loss = 0.0
                val_samples = 0
                for x_batch, *_ in val_loader:
                    x_batch = x_batch.to(device)
                    val_loss += svi.evaluate_loss(x_batch)
                    val_samples += x_batch.size(0)
                
                avg_val_loss = val_loss / val_samples
            
            fold_losses.append(avg_val_loss)
            print(f"  Fold {fold_idx+1} - Loss validação: {avg_val_loss:.2f}")
            
            # Reportar valor intermediário ao Optuna (para pruning)
            trial.report(avg_val_loss, fold_idx)
            
            # Verificar se deve fazer pruning (parar early)
            if trial.should_prune():
                print(f"  Trial {trial.number + 1} foi podado (pruned)!")
                raise optuna.TrialPruned()
        
        # Usar mediana como métrica
        median_loss = np.median(fold_losses)
        print(f"→ Loss mediana CV: {median_loss:.4f}")
        
        return median_loss
    
    # Criar study do Optuna
    print(f"\n{'#'*60}")
    print(f"INICIANDO OTIMIZAÇÃO COM OPTUNA ({n_trials} trials)")
    print(f"{'#'*60}\n")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    study.optimize(
        objective, 
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Extrair melhores resultados
    best_params = study.best_params
    best_loss = study.best_value
    
    print(f"\n{'#'*60}")
    print(f"MELHOR CONFIGURAÇÃO ENCONTRADA:")
    print(f"{'#'*60}")
    print(f"Parâmetros: {best_params}")
    print(f"Loss: {best_loss:.4f}")
    print(f"Trial número: {study.best_trial.number + 1}")
    print(f"Trials completados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Trials podados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"{'#'*60}\n")
    
    return best_params, best_loss, study