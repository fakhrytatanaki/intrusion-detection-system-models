import numpy as np



class HistogramDist:
    def __init__(self,include_max_margin=1e-5,quantile_bounds=(0,1)):
        self.include_max_margin = include_max_margin
        self.quantile_bounds = quantile_bounds
        assert(quantile_bounds[1] > quantile_bounds[0])

    def fit(self,data,num_bins=100,interval=None):

        self.d_min = np.quantile(data,self.quantile_bounds[0])
        self.d_max = np.quantile(data,self.quantile_bounds[1]) + self.include_max_margin

        if interval:
            self.num_bins=int((self.d_max - self.d_min)//interval)
        else:
            self.num_bins = num_bins


        self.num_samples = 0
        self.dist_count = np.zeros(self.num_bins,dtype=np.int64)

        self.samples_range = np.linspace(
                self.d_min,
                self.d_max,
                num=self.num_bins
                )

        self.diff = (self.d_max - self.d_min)/self.num_bins


        for x in data:
            if self.d_min < x < self.d_max: # don't include outliers
                self.dist_count[int(np.floor((x-self.d_min)/self.diff))]+=1

        self.num_samples+=len(data)
        self.dist_prob = self.dist_count/self.num_samples
        return self


    def probability_where_value_belongs(self,x):
        if not self.d_min <= x <= self.d_max:
            return 0 #impossible (not part of the distribution)

        return self.dist_prob[int(np.floor((x-self.d_min)/self.diff))]

    def expected_val(self):
        return np.sum(self.samples_range*self.dist_prob)


        return p
    def probability_range(self,x_min,x_max):
        p = 0

        ind_begin=max( int(np.floor((x_min-self.d_min)/self.diff)), 0)

        ind_end=min( int(np.floor((x_max-self.d_min)/self.diff)), self.num_bins-1)


        for i in range(ind_begin,ind_end+1):
           p+=self.dist_prob[i]


        return p
