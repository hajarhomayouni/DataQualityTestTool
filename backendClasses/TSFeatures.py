import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

class TSFeatures:

    def extract_features(self, timeseries):
        oddstream=importr('oddstream')

        #r_timeseries = pandas2ri.py2ri(timeseries)
        with localconverter(ro.default_converter + pandas2ri.converter):
            print("timeseries")
            print(timeseries)
            r_timeseries = ro.conversion.py2rpy(timeseries)
            features=oddstream.extract_tsfeatures(r_timeseries)
            features= ro.conversion.rpy2py(features)
            print("features")
            print(features)
            print(features.shape)
            return features
        return []
