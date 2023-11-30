
import os, sys
import motmot.FlyMovieFormat.FlyMovieFormat as fmf
import motmot.ufmf.ufmf as ufmf

import copy, numpy as np

#
# The 'SpiderMovie' class hides the properties of the underlying behavioral recording file
# from the user, and returns uncompressed frames, regardless of the underlying compression.
#

class SpiderMovie():
    def __init__(self, fname = None):
        # Set attributes to defaults
        self.dtype = np.uint8
        self.ndim = 0
        self.shape = ()
        self.min = 0
        self.max = 0
        self.size = 0

        self._reader, self._bg, self.mask = None, None, None

        # Read file if specified
        if fname is not None:
            self._reader = None
            self._bg = None

            # Handle .ufmf files
            if fname.lower().endswith('.ufmf'):
                # Check if this UFMF has background subtracted
                try:
                    _r = ufmf.UfmfV3(fname)
                    self._bg = _r.get_keyframe_for_timestamp('bg', timestamp=0)[0]
                    _r.close()
                except:
                    self._bg = None
                # Get reader
                self._reader = ufmf.FlyMovieEmulator(fname)

            elif fname.lower().endswith('.fmf'):
                # Get reader
                self._reader = fmf.FlyMovie(fname)

            # Set attributes
            try:
                self.dtype = self.get_frame(0)[0].dtype
                self.shape = tuple([len(self),] + list(self.get_frame(0)[0].shape))
                self.ndim = len(self.shape)
                self.size = np.prod(np.array(self.shape, dtype=np.int64))
            except:
                pass

            # Create mask (supports indexing)
            self.mask = [np.arange(self.shape[i], dtype=np.int64) for i in range(self.ndim)]

    # Return the number of frames, unless this movie has been subsetted
    def __len__(self):
        if self.mask is None:
            return self._reader.get_n_frames()
        else:
            return self.shape[0]

    # Overrides standard copying behavior to avoid issues with the standard copy behavior
    def __copy__(self):
        n = SpiderMovie()
        n._bg = self._bg
        n._reader = self._reader

        n.dtype = self.dtype
        n.ndim = self.ndim
        n.shape = self.shape
        n.min = self.min
        n.max = self.max
        n.size = self.size

        n.mask = [x for x in self.mask]

        return n

    def __getitem__(self, sliced):
        # Return a subset of the movie volume
        if isinstance(sliced, int) or isinstance(sliced, np.int32) or isinstance(sliced, np.int64):
            f = self.get_frame(self.mask[0][sliced])[0]
            return f[np.ix_(self.mask[1], self.mask[2])]
        # Slice each of the axes
        elif isinstance(sliced, list):
            # Update mask, shape, ndim, size
            n = copy.copy(self)
            nshape = list(n.shape)
            for i in range(len(sliced)):
                n.mask[i] = n.mask[i][sliced[i]]
                nshape[i] = len(n.mask[i])
            n.shape = tuple(nshape)
            n.size = np.prod(np.array(n.shape, dtype=np.int64))
            return n
        else:
            raise NotImplementedError("Invalid index for SpiderMovie")

    # This function is added for PyQtGraph
    def implements(self, x):
        if x == 'MetaArray':
            return True
        else:
            return False

    # This class implements most array functionality, and a reference to self is therefore returned.
    def asarray(self):
        return self

    # We define transpose() b/c some libraries downstream require it to exist.
    # As a quick fix, we simply return the un-transposed matrix...
    # In the future, actually return a transposed image
    def transpose(self, axorder):
        return self

    # 'Override' the get_frame function provided by the FlyMovieFormat classes
    def get_frame(self, fidx):
        try:
            frame, ts = self._reader.get_frame(fidx)
            if self._bg is not None:
                frame += self._bg
            return frame, ts
        except Exception as e:
            if 'short frame' in str(e):
                return np.zeros(self.shape[1:], dtype=np.uint8), -1
            else:
                raise e

    # Functions that have not been 'overridden' are called in the 'parent' (_reader) object.
    def __getattr__(self, name):
        # NumPy requires these attributes are present (but can be undefined) to treat this instance as an array.
        if name == '__array_struct__':
            raise AttributeError()
        # NumPy requires these attributes are present (but can be undefined) to treat this instance as an array.
        elif name == '__array_interface__':
            raise AttributeError()
        # Obtain any other attributes from the underlying reader instance
        else:
            return getattr(self._reader, name)