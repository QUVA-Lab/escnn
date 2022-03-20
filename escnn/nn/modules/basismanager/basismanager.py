
from abc import ABC, abstractmethod

from typing import List, Iterable, Dict, Union


__all__ = ["BasisManager"]


class BasisManager(ABC):
    
    def __init__(self):
        r"""
        Abstract class defining the *interface* for the different modules which deal with the filter basis.
        It provides a few methods which can be used to retrieve information about the basis and each of its elements.
       
        """
        super(BasisManager, self).__init__()
        
    @abstractmethod
    def get_element_info(self, name: int) -> Dict:
        """
        Method that returns the information associated to a basis element
        
        Parameters:
            name (int): index of the basis element
        
        Returns:
            dictionary containing the information
        """
        pass
    
    @abstractmethod
    def get_basis_info(self) -> Iterable[Dict]:
        """
        Method that returns an iterable over all basis elements' attributes.

        Returns:
            an iterable over all the basis elements' attributes
            
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        r"""
        The dimensionality of the basis and, so, the number of weights needed to expand it.
        
        Returns:
            the dimensionality of the basis
            
        """
        pass


