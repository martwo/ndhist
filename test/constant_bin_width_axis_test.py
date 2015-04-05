import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_constant_bin_width_axis(self):
        """Tests if the constant_bin_width_axis class works properly.
        """
        import math

        stop = 10
        start = 0
        width = 1
        axis = ndhist.axes.linear(start,stop,width, label='MyLabel', name='MyAxis')

        def _check_axis(axis):
            self.assertTrue(axis.name == 'MyAxis')
            self.assertTrue(axis.label == 'MyLabel')
            # The name and label is changeable. So lets try to change it.
            axis.name = 'MyNewAxis'
            self.assertTrue(axis.name == 'MyNewAxis')
            axis.label = 'MyNewLabel'
            self.assertTrue(axis.label == 'MyNewLabel')

            # Change back the label and name.
            axis.name = 'MyAxis'
            self.assertTrue(axis.name == 'MyAxis')
            axis.label = 'MyLabel'
            self.assertTrue(axis.label == 'MyLabel')

            # The dtype of the axis is choosen automatically by the numpy.linspace
            # function.
            nbins = int(math.ceil((stop - start) / width)) + 1
            edges_dtype = np.linspace(start, stop, num=nbins, endpoint=True).dtype
            self.assertTrue(axis.dtype == edges_dtype)

            self.assertTrue(axis.has_underflow_bin)

            self.assertTrue(axis.has_overflow_bin)

            self.assertFalse(axis.is_extendable)

            self.assertTrue(axis.nbins == 12)

            self.assertTrue(np.all(axis.binedges == np.array(
                [-np.inf,0,1,2,3,4,5,6,7,8,9,10,+np.inf]
            )))

            self.assertTrue(np.all(axis.lower_binedges == np.array(
                [-np.inf,0,1,2,3,4,5,6,7,8,9,10]
            )))

            self.assertTrue(np.all(axis.upper_binedges == np.array(
                [0,1,2,3,4,5,6,7,8,9,10,+np.inf]
            )))

            self.assertTrue(np.all(axis.bincenters == np.array(
                [-np.inf,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,+np.inf]
            )))

            self.assertTrue(np.all(axis.binwidths == np.array(
                [np.inf,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,np.inf]
            )))

        _check_axis(axis)

        # Put the axis into a ndhist object and check the values again.
        h = ndhist.ndhist((axis,))
        _check_axis(h.axes[0])

if(__name__ == "__main__"):
    unittest.main()
