import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogDecorator:
  
  def __init__(self, name):
    self.__name = name
    self.__path = 'features/{}.csv'.format(self.__name)
  
  def __call__(self, f, *args, **kwargs):
    if not os.path.exists(self.__path):
      logger.info('Running %s feature selection...', self.__name)
      start = time.time()
    else:
      logger.info('Reading %s features from file : [%s]...', self.__name, self.__path)
    
    def new_f(*args, **kwargs):
      X = f(*args, **kwargs)
      if not os.path.exists(self.__path):
        logger.info('Saving %s features to [%s]...', self.__name, self.__path)
        logger.info('%s selection last [%s]...', self.__name, (time.time() - start) * 1000)
      
      else:
        logger.info('Read %s features from file : [%s]...', self.__name, self.__path)
      return X
    
    new_f.__name__ = f.__name__
    return new_f
