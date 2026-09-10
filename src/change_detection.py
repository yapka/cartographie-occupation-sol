"""
Module de détection de changements d'occupation du sol
Analyse les différences entre deux périodes
"""

import numpy as np
from scipy import ndimage


class ChangeDetector:
    """Détecteur de changements d'occupation du sol"""
    
    def __init__(self):
        self.t1_classification = None
        self.t2_classification = None
        self.change_map = None
        
    def detect_changes(self, classification_t1, classification_t2):
        """
        Détecter les changements entre deux dates
        
        Parameters
        ----------
        classification_t1 : array
            Classification à t1
        classification_t2 : array
            Classification à t2
            
        Returns
        -------
        change_map : array
            Carte des changements (0=pas de changement, 1=changement)
        """
        self.t1_classification = classification_t1
        self.t2_classification = classification_t2
        
        self.change_map = (classification_t1 != classification_t2).astype(int)
        
        return self.change_map
    
    def transition_matrix(self, classification_t1, classification_t2):
        """
        Calculer la matrice de transition
        
        Parameters
        ----------
        classification_t1, classification_t2 : arrays
            Classifications aux deux dates
            
        Returns
        -------
        transition : dict
            Matrice de transition avec changements
        """
        classes_t1 = np.unique(classification_t1)
        classes_t2 = np.unique(classification_t2)
        
        transition = {}
        
        for c1 in classes_t1:
            mask_t1 = classification_t1 == c1
            transitions_from_c1 = classification_t2[mask_t1]
            
            for c2 in classes_t2:
                count = np.sum(transitions_from_c1 == c2)
                if count > 0:
                    key = f"{c1} → {c2}"
                    transition[key] = count
        
        return transition
    
    def connectivity_analysis(self, change_map, min_size=100):
        """
        Analyser la connectivité des zones de changement
        
        Parameters
        ----------
        change_map : array
            Carte des changements binaire
        min_size : int
            Taille minimale des régions en pixels
            
        Returns
        -------
        labeled : array
            Carte avec régions étiquetées
        stats : dict
            Statistiques des régions
        """
        # Étiqueter les régions connectées
        labeled, num_features = ndimage.label(change_map)
        
        stats = {
            'num_regions': num_features,
            'total_pixels': np.sum(change_map),
            'average_size': np.sum(change_map) / num_features if num_features > 0 else 0,
            'regions': {}
        }
        
        # Statistiques par région
        for i in range(1, num_features + 1):
            region = labeled == i
            size = np.sum(region)
            
            if size >= min_size:
                stats['regions'][i] = {
                    'size_pixels': int(size),
                    'centroid': ndimage.center_of_mass(region)
                }
        
        return labeled, stats
    
    def compute_statistics(self, classification_t1, classification_t2, 
                          class_names=None, pixel_area_m2=100):
        """
        Calculer les statistiques d'occupation du sol
        
        Parameters
        ----------
        classification_t1, classification_t2 : arrays
            Classifications aux deux dates
        class_names : dict
            Mapping classe -> nom
        pixel_area_m2 : float
            Aire d'un pixel en m²
            
        Returns
        -------
        stats : dict
            Statistiques par classe
        """
        if class_names is None:
            classes = np.unique(np.concatenate([classification_t1, classification_t2]))
            class_names = {c: f"Class_{c}" for c in classes}
        
        stats = {}
        
        for class_id, class_name in class_names.items():
            count_t1 = np.sum(classification_t1 == class_id)
            count_t2 = np.sum(classification_t2 == class_id)
            
            area_t1_ha = count_t1 * pixel_area_m2 / 10000
            area_t2_ha = count_t2 * pixel_area_m2 / 10000
            
            change_ha = area_t2_ha - area_t1_ha
            change_pct = (change_ha / area_t1_ha * 100) if area_t1_ha > 0 else 0
            
            stats[class_name] = {
                'pixels_t1': int(count_t1),
                'pixels_t2': int(count_t2),
                'area_ha_t1': area_t1_ha,
                'area_ha_t2': area_t2_ha,
                'change_ha': change_ha,
                'change_percent': change_pct
            }
        
        return stats


class SpectralChangeDetector:
    """Détection de changements basée sur indices spectraux"""
    
    @staticmethod
    def ndvi_change(ndvi_t1, ndvi_t2, threshold=0.1):
        """
        Détecter les changements de végétation
        
        Parameters
        ----------
        ndvi_t1, ndvi_t2 : arrays
            NDVI aux deux dates
        threshold : float
            Seuil de changement
            
        Returns
        -------
        change : array
            Pixels de changement
        """
        change = np.abs(ndvi_t2 - ndvi_t1) > threshold
        return change.astype(int)
    
    @staticmethod
    def urban_expansion(ndbi_t1, ndbi_t2, threshold=0.1):
        """
        Détecter l'expansion urbaine
        
        Parameters
        ----------
        ndbi_t1, ndbi_t2 : arrays
            NDBI aux deux dates
        threshold : float
            Seuil d'urbanisation
            
        Returns
        -------
        expansion : array
            Zones d'expansion urbaine
        """
        # Nouvelles zones urbaines
        urban_t1 = ndbi_t1 > 0.05
        urban_t2 = ndbi_t2 > 0.05
        
        expansion = (urban_t2 & ~urban_t1).astype(int)
        
        return expansion
    
    @staticmethod
    def deforestation(ndvi_t1, ndvi_t2, threshold=0.3):
        """
        Détecter la déforestation
        
        Parameters
        ----------
        ndvi_t1, ndvi_t2 : arrays
            NDVI aux deux dates
        threshold : float
            Seuil de déforestation
            
        Returns
        -------
        deforestation : array
            Zones déforestées
        """
        forest_t1 = ndvi_t1 > threshold
        forest_t2 = ndvi_t2 > threshold
        
        deforestation = (forest_t1 & ~forest_t2).astype(int)
        
        return deforestation
