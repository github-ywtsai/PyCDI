""" dynamic plot """

import matplotlib.pyplot as plt
import numpy as np


class PlotLogger(object):
    """Object that represents a logger to track image difference between iterations.
    
    :basic class for using logger
    
    """

    def __init__(self, block = False, niter = None, savelast = False, DyPlOnOff = False):
        """
        :param block: block the logger in the last iteration. Default: block = False
        :param niter: numbers of iteration. Notice: this parameter must be used together with block = True. Default: None
        :param savelast: save the last logger (figure.png) in work directory. Default: False
        :param DyPlOnOff: DynamicPlot on/off. To check method, dynamicPlot, that 1st run.
        
        """
        self.block = block
        self.niter = niter
        self.acc_DyPl = 0 # to show the nth iteration for dynamicPlot
        self.savelast = savelast
        self.DyPlOnOff = DyPlOnOff
        self.fig = None

    
    def dynamicPlot(self, *args, **kwargs):

        self.nargs = len(args)
        self.args = list(args)
        self.nkwargs = len(kwargs)
        self.kwargs = list(kwargs.values())

        plt.ion()

        if self.DyPlOnOff == False:
            self.fig = plt.figure()
            self.DyPlOnOff = True
            print("Plot Logger starts")
        else:
            self.DyPlOnOff = True
            

        self.ax = []
        self.img = []

        for i in range(self.nargs):
            self.ax.append(plt.subplot(1,self.nargs,int(i+1)))
            self.img.append(plt.imshow(self.args[i]))
            if self.nkwargs != 0:
                plt.title(self.kwargs[i])
            elif self.nkwargs == 0:
                pass
        plt.suptitle('iter: %d'%(self.acc_DyPl))


        if self.savelast == True:
            self.fig.savefig('./log_last_plot.png')


        if self.niter == None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.clf()
        elif self.niter != None:
            if self.block == True:
                if self.acc_DyPl != (self.niter-1):
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    plt.clf()
                elif self.acc_DyPl == (self.niter-1):
                    plt.ioff()
                    plt.show(block = True)
            else:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.clf()
        self.acc_DyPl += 1


    def Resume(self):
        self.block = False
        self.niter = None
        self.acc_DyPl = 0
        self.savelast = False
        self.DyPlOnOff = False
        self.fig = None


