from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_foundation.components.abstract_component import AbstractComponent


class FeatureCatalog(AbstractComponent):

    def __init__(self, property_manager: FeatureCatalogPropertyManager, intent_model: FeatureCatalogIntentModel,
                 default_save=None):
        """ Encapsulation class for the discovery set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:

                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save)

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, default_save=None, **kwargs):
        """ Class Factory Method to instantiates the component application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.

        by default the handler is local Pandas but also supports remote AWS S3 and Redis. It use these Factory
        instantiations ensure that the schema is s3:// or redis:// and the handler will be automatically redirected

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param kwargs: to pass to the connector contract
         :return: the initialised class instance
         """
        _pm = FeatureCatalogPropertyManager(task_name=task_name)
        _intent_model = FeatureCatalogIntentModel(property_manager=_pm)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, **kwargs)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save)

    @classmethod
    def _from_remote_s3(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers an Amazon AWS s3 remote store."""
        _module_name = 'ds_connectors.handler.aws_s3_handlers'
        _handler = 'AwsS3PersistHandler'
        return _module_name, _handler

    @classmethod
    def _from_remote_redis(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers an Amazon AWS s3 remote store."""
        _module_name = 'ds_connectors.handlers.redis_handlers'
        _handler = 'RedisPersistHandler'
        return _module_name, _handler
