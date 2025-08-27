#!/usr/bin/env python3

from infrastructure.repositories.redis_user_repository import RedisUserRepository
from infrastructure.repositories.file_dataset_repository import FileDatasetRepository
from services.application.user_service import UserService
from services.application.dataset_service import DatasetService
from presentation.controllers.user_controller import UserController, AdminController
from presentation.controllers.dataset_controller import DatasetController


class Container:
    """Dependency injection container"""
    
    def __init__(self):
        self._repositories = {}
        self._services = {}
        self._controllers = {}
        self._setup_dependencies()
    
    def _setup_dependencies(self):
        """Setup all dependencies with proper injection"""
        
        # Repositories
        self._repositories['user'] = RedisUserRepository()
        self._repositories['dataset'] = FileDatasetRepository()
        
        # Application Services
        self._services['user'] = UserService(self._repositories['user'])
        self._services['dataset'] = DatasetService(self._repositories['dataset'])
        
        # Controllers
        self._controllers['user'] = UserController(self._services['user'])
        self._controllers['admin'] = AdminController(self._services['user'])
        self._controllers['dataset'] = DatasetController(
            self._services['dataset'], 
            self._services['user']
        )
    
    def get_repository(self, name: str):
        """Get repository by name"""
        return self._repositories.get(name)
    
    def get_service(self, name: str):
        """Get service by name"""
        return self._services.get(name)
    
    def get_controller(self, name: str):
        """Get controller by name"""
        return self._controllers.get(name)
    
    def get_all_routers(self):
        """Get all FastAPI routers from controllers"""
        routers = []
        for controller in self._controllers.values():
            if hasattr(controller, 'router'):
                routers.append(controller.router)
        return routers


# Global container instance
container = Container()