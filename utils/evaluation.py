import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from typing import Dict, List

class HSIEvaluation:
    def __init__(self, param: Dict):
        self.param = param
        self.target_names = self.get_target_names()
        self.res = {}

    def get_target_names(self) -> List[str]:
        data_sign = self.param['data']['data_sign']
        if data_sign == 'IndianPine':  
            return ['Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 
                     'Hay-windrowed', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers','Alfalfa','Grass-pasture-mowed', 'Oats']
        elif data_sign == "Pavia":
            return ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted_metal_sheets', 'Bare_Soil', 
                    'Bitumen', 'Self_Blocking_Bricks', 'Shadows']
        elif data_sign == "Houston":
            return ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 
                    'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1', 
                    'Parking Lot 2', 'Tennis Court', 'Running Track']
        elif data_sign == "Salinas":
            return ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth',
                    'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                    'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                    'Vinyard_untrained', 'Vinyard_vertical_trellis']
        elif data_sign == "WHHH":
            return ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton Firewood', 'Rape', 'Chinese cabbage', 'Pakchoi', ' Cabbage', 'Tuber mustard',
                    'Brassica Parachinensis', 'Brassica chinensism', 'Small Brassica Chinensis', 'Latuca Sativa', 'Celtuce', 'Film Covered Lettuce', 'Romaine Lettuce',
                    'Carrot', 'White Radish', 'Garlic sprout', 'Broad bean', 'Tree']
        elif data_sign == "KSC":
            return ['Scrub', 'Willow Swamp', 'CP hammock', 'Slash pine', 'Oak-Broadleaf', 'Hardwood', 'Swap', 
                    'Graminoid marsh', 'Spartina marsh', 'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water']
        elif data_sign == "Plastic":
            return ['Fillet', 'ABS', 'FAB', 'HDPE', 'LDPE', 'NYL', 'PET', 'PP', 'PS', 'PUR', 'PVC', 'RUB', 'TEF']
        else:
            return None

    def AA_andEachClassAccuracy(self, confusion_matrix: np.ndarray) -> tuple:
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(list_diag / list_raw_sum)
        average_acc = np.mean(each_acc)
        return each_acc, average_acc


    
    def eval(self, y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict:
        class_num = np.max(y_test) + 1
        classification = classification_report(y_test, y_pred_test, 
                labels=list(range(class_num)), digits=4, target_names=self.target_names, zero_division=0,output_dict=True)
        oa = accuracy_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test, labels=list(range(class_num)))
        each_acc, aa = self.AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred_test)

        
        self.res['oa'] = oa * 100
        self.res['aa'] = aa * 100
        self.res['kappa'] = kappa * 100
        self.res['each_acc'] = (each_acc * 100).tolist()
        self.res['confusion'] = format_confusion_matrix(confusion.tolist())
        self.res['classification'] = classification
        

        return self.res

class PaviaCenterEvaluation(HSIEvaluation):
    """Special evaluation class for PaviaCenter that handles missing classes correctly"""
    
    def eval(self, y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict:
        # Get only the classes that actually exist in the data
        unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
        
        # Get target names for existing classes only
        all_pavia_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted_metal_sheets', 'Bare_Soil', 
                          'Bitumen', 'Self_Blocking_Bricks', 'Shadows']
        
        existing_target_names = [all_pavia_names[i] for i in unique_labels]
        
        # Classification report with only existing labels
        classification = classification_report(y_test, y_pred_test, 
                labels=unique_labels, digits=4, target_names=existing_target_names, zero_division=0, output_dict=True)
        
        # Overall accuracy
        oa = accuracy_score(y_test, y_pred_test)
        
        # Confusion matrix with only existing classes
        confusion = confusion_matrix(y_test, y_pred_test, labels=unique_labels)
        
        # Average accuracy - only for existing classes
        each_acc, aa = self.AA_andEachClassAccuracy(confusion)
        
        # Kappa
        kappa = cohen_kappa_score(y_test, y_pred_test)
        
        # Build results with proper mapping to all 9 Pavia classes for display
        # Initialize with zeros for missing classes
        full_each_acc = [0.0] * 9
        for i, label in enumerate(unique_labels):
            full_each_acc[label] = each_acc[i] * 100
        
        self.res['oa'] = oa * 100
        self.res['aa'] = aa * 100  # This is the correct AA - only averaging existing classes
        self.res['kappa'] = kappa * 100
        self.res['each_acc'] = full_each_acc  # Full 9-class array with zeros for missing classes (for display compatibility)
        self.res['confusion'] = format_confusion_matrix(confusion.tolist())
        self.res['classification'] = classification
        self.res['existing_classes'] = unique_labels  # Track which classes exist
        self.res['num_classes_evaluated'] = len(unique_labels)  # Track how many classes were actually evaluated
        self.res['samples'] = len(y_test)  # Add sample count
        
        return self.res

def format_confusion_matrix(confusion_matrix):
    formatted_rows = []
    for row in confusion_matrix:
        formatted_row = '    ' + str(row)
        formatted_rows.append(formatted_row)
    formatted_matrix = '[\n' + ',\n'.join(formatted_rows) + '\n]'
    return formatted_matrix