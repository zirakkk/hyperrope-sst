import torch
import torch.nn as nn
import torch.optim as optim
from models.hyperrope_vit import HyperRopeViT
from models.lsga_vit import LSGA_VIT
from models.hit import HiT
from models.sqsformer import SQSFormer
from models.spectralformer import SpectralFormer
from models.conv2d import Conv2d
from models.conv3d import Conv3d
from models.ssrn import SSRN
from utils.utils import device, recorder
from utils.evaluation import HSIEvaluation, PaviaCenterEvaluation
import numpy as np
from typing import Dict, Tuple
import random
from configs.config import CHECKPOINT_PATH_PREFIX


class BaseTrainer:
    def __init__(self, params: Dict):
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = device
        # Use PaviaCenterEvaluation for PaviaCenter data to handle missing classes correctly
        if params['data']['data_sign'] == 'PaviaCenter':
            self.evaluator = PaviaCenterEvaluation(param=params)
        else:
            self.evaluator = HSIEvaluation(param=params)

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = self.train_params.get('clip', 15)

       
    def train(self, train_loader, valid_loader=None):
        #torch.autograd.set_detect_anomaly(True)
        epochs = self.params['train'].get('epochs', 200)
        best_valid_oa = 0
        patience = self.params['train'].get('patience', 10)
        no_improve_count = 0

        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            recorder.append_index_value("epoch_loss", epoch + 1, avg_loss)
            print(f'[Epoch: {epoch + 1}] [Loss: {avg_loss:.5f}]')

            # Adaptive validation for Salinas dataset only
            if valid_loader is not None and self.params['data']['data_sign'] == 'Salinas':
                # Validate every 5 epochs early, every 3 epochs mid, every 2 epochs late, plus first/last epochs
                should_validate = (epoch < 3 or epoch >= epochs - 3 or 
                                 (epoch <= 30 and (epoch + 1) % 5 == 0) or
                                 (31 <= epoch <= 100 and (epoch + 1) % 3 == 0) or
                                 (epoch > 100 and (epoch + 1) % 2 == 0))
                
                if should_validate:
                    print(f'[Validating Salinas at epoch {epoch + 1}]')
                    valid_oa = self.validate(valid_loader, epoch)
                    if valid_oa > best_valid_oa:
                        best_valid_oa = valid_oa
                        self.save_checkpoint(epoch, best_valid_oa, is_best=True)  # Save best model
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= patience:
                            print(f'Early stopping after {epoch + 1} epochs')
                            break
            # Original validation for other datasets (except WHHH)
            elif valid_loader is not None and (epoch+1) % 1 == 0 and self.params['data']['data_sign'] not in ['WHHH', 'Salinas']:
                valid_oa = self.validate(valid_loader, epoch)
                if valid_oa > best_valid_oa:
                    best_valid_oa = valid_oa
                    self.save_checkpoint(epoch, best_valid_oa, is_best=True)  # Save best model
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f'Early stopping after {epoch + 1} epochs')
                        break
            elif valid_loader is not None and (epoch+1) % 10 == 0 and self.params['data']['data_sign'] == 'WHHH':
                valid_oa = self.validate(valid_loader, epoch)
                if valid_oa > best_valid_oa:
                    best_valid_oa = valid_oa
                    self.save_checkpoint(epoch, best_valid_oa, is_best=True)  # Save best model
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f'Early stopping after {epoch + 1} epochs')
                        break
                    
        self.save_checkpoint(epoch, best_valid_oa, is_best=False)  # Save last model
        print('Finished Training')
        return True

    def validate(self, valid_loader, epoch):
        self.net.eval()
        y_pred_valid, y_valid = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)
                outputs = outputs.detach().cpu().numpy()
                y_pred_valid.append(np.argmax(outputs, axis=1))
                y_valid.append(labels.numpy())
        
        y_pred_valid = np.concatenate(y_pred_valid)
        y_valid = np.concatenate(y_valid)
        temp_res = self.evaluator.eval(y_valid, y_pred_valid)
        
        recorder.append_index_value("valid_oa", epoch+1, temp_res['oa'])
        recorder.append_index_value("valid_aa", epoch+1, temp_res['aa'])
        recorder.append_index_value("valid_kappa", epoch+1, temp_res['kappa'])
        print(f'[Validation] [Epoch: {epoch+1}] [OA: {temp_res["oa"]:.5f}] [AA: {temp_res["aa"]:.5f}] [Kappa: {temp_res["kappa"]:.5f}]')
        
        return temp_res['oa']

    def save_checkpoint(self, epoch, valid_oa, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'valid_oa': valid_oa,
        }
        
        if is_best:
            torch.save(checkpoint, f"{CHECKPOINT_PATH_PREFIX}/{self.params['data']['data_sign']}_best_model_{self.params['net']['trainer']}.pth")
            print(f'Best model checkpoint saved at epoch {epoch}')
        #else:
        #    torch.save(checkpoint, f"{CHECKPOINT_PATH_PREFIX}/{self.params['data']['data_sign']}_last_model_{self.params['net']['trainer']}.pth")
        #    print(f'Last model checkpoint saved at epoch {epoch}')


    def final_eval(self, test_loader, checkpoint_path=None):
        if checkpoint_path==None:
            checkpoint = torch.load(f"{CHECKPOINT_PATH_PREFIX}/{self.params['data']['data_sign']}_best_model_{self.params['net']['trainer']}.pth")
        else:
            checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded from epoch {checkpoint['epoch']} with OA {checkpoint['valid_oa']}")
        y_pred_test, y_test = self.test(test_loader)
        return self.evaluator.eval(y_test, y_pred_test), y_test, y_pred_test


    def test(self, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        y_pred_test = []
        y_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)
                outputs = outputs.detach().cpu().numpy()
                y_pred_test.append(np.argmax(outputs, axis=1))
                y_test.append(labels.numpy())
        
        return np.concatenate(y_pred_test), np.concatenate(y_test)

class HyperRopeViTTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(HyperRopeViTTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.net = HyperRopeViT(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    
    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
    
    def train(self, train_loader, valid_loader=None):
        result = super().train(train_loader, valid_loader)
#        if valid_loader:
#           self.scheduler.step(self.best_valid_oa)
        return result
    
class LSGAViTTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(LSGAViTTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.net = LSGA_VIT(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)

class HiTTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(HiTTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.net = HiT(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
    
class SpectralFormerTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(SpectralFormerTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.net = SpectralFormer(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.0005)
        self.weight_decay = self.train_params.get('weight_decay', 0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)

class SQSFormerTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(SQSFormerTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.net = SQSFormer(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        logits = outputs
        return self.criterion(logits, target)        

class Conv2dTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(Conv2dTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.net = Conv2d(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)

class Conv3dTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(Conv3dTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.net = Conv3d(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)

class SSRNTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(SSRNTrainer, self).__init__(params)
        
        self.data = params['data']
        random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])
        torch.cuda.manual_seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Update params with dataset-specific k value for SSRN
        if 'k' in self.data:
            params['net']['k'] = self.data['k']
        
        self.net = SSRN(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        logits = outputs
        if len(logits.shape) == 1:
            logits = torch.unsqueeze(logits, 0)
        return self.criterion(logits, target)



def get_trainer(params: Dict):
    trainer_type = params['net']['trainer']
    if trainer_type == "hyperrope_vit":
        return HyperRopeViTTrainer(params)
    elif trainer_type == "lsga_vit":
        return LSGAViTTrainer(params)
    elif trainer_type == "hit":
        return HiTTrainer(params)
    elif trainer_type == "spectralformer":
        return SpectralFormerTrainer(params)
    elif trainer_type == "sqsformer":
        return SQSFormerTrainer(params)
    elif trainer_type == "conv2d":
        return Conv2dTrainer(params)
    elif trainer_type == "conv3d":
        return Conv3dTrainer(params)
    elif trainer_type == "ssrn":
        return SSRNTrainer(params)
    raise Exception(f"Trainer not implemented for {trainer_type}")