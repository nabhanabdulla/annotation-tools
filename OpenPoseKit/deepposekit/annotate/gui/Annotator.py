# -*- coding: utf-8 -*-
# Copyright 2018-2019 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import h5py
import os

from deepposekit.annotate.gui.GUI import GUI
from deepposekit.annotate.utils import hotkeys as keys

__all__ = ["Annotator"]


class Annotator(GUI):

    """
    A GUI for annotating images.

    ------------------------------------------------------------
         Keys             |   Action
    ------------------------------------------------------------
    >    +,-              |   Rescale the image
    >    Left mouse       |   Move active keypoint
    >    W, A, S, D       |   Move active keypoint
    >    space            |   Changes W,A,S,D mode (swaps between 1px or 10px)
    >    J, L             |   Load previous or next image
    >    <, >             |   Jump 10 images backward or forward
    >    I, K or          |
         tab, shift+tab   |   Switch active keypoint
    >    R                |   Mark frame as unannotated, or "reset"
    >    F                |   Mark frame as annotated or "finished"
    >    Esc, Q           |   Quit the Annotator GUI
    ------------------------------------------------------------

    Note: Data is automatically saved when moving between frames.

    Parameters
    ----------
    datapath: str
        Filepath of the HDF5 (.h5) file that contains the images to
        be annotated.

    dataset: str
        Key name to access the images in the .h5 file.

    skeleton: str
        Filepath of the .csv or .xlsx file that has indexed information
        on name of the keypoint (part, e.g. head), parent (the direct
        connecting part, e.g. neck connects to head, parent is head),
        and swap (swapping positions with a part when reflected).

        See example file for more information.

    scale: int/float, default 1
        Scaling factor for the GUI (e.g. used in zooming).

    text_scale: float
        Scaling factor for the GUI font.
        A text_scale of 1 works well for 1920x1080 (1080p) images
    
    shuffle_colors: bool, default = True
        Whether to shuffle the color order for drawing keypoints

    refresh: int, default 100
        Delay on receiving next keyboard input in milliseconds.

    Attributes
    ----------
    window_name: str
        Name of the Annotation window when running program.
        Set to be 'Annotation' unless otherwise changed.

    n_images: int
        Number of images in the .h5 file.

    n_keypoints: int
        Number of keypoints in the skeleton.

    key: int
        The key that is pressed on the keyboard.

    image_idx: int
        Index of a specific image in the .h5 file.

    image: numpy.ndarray
        One image accessed using image_idx.

    Example
    -------
    >>> from deepposekit import Annotator
    >>> app = Annotator('annotation.h5', 'images', 'skeleton.csv')
    >>> app.run()

    """

    def __init__(
        self,
        datapath,
        dataset,
        skeleton,
        scale=1,
        text_scale=0.15,
        shuffle_colors=True,
        refresh=100,
    ):
        super(GUI, self).__init__()
        
        self.window_name = "Annotation"
        self.shuffle_colors = shuffle_colors
        self._init_skeleton(skeleton)
        if os.path.exists(datapath):
            self._init_data(datapath, dataset)
        else:
            raise ValueError("datapath file or path does not exist")
        self._init_gui(scale, text_scale, shuffle_colors, refresh)

    def _init_data(self, datapath, dataset):
        """ Initializes the images from the .h5 file (called in init).

        Parameters
        ----------
        datapath: str
            Path of the .h5 file that contains the images to be annotated.

        dataset: str
            Key name to access the images in the .h5 file.

        """
        if isinstance(datapath, str):
            if datapath.endswith(".h5"):
                self.datapath = datapath
            else:
                raise ValueError("datapath must be .h5 file")
        else:
            raise TypeError("datapath must be type `str`")

        if isinstance(dataset, str):
            self.dataset = dataset
        else:
            raise TypeError("dataset must be type `str`")

        with h5py.File(self.datapath, "r+") as h5file:

            self.n_images = h5file[self.dataset].shape[0]
            # Check that all parts of the file exist
            if "annotations" not in list(h5file.keys()):
                empty_array = np.zeros((self.n_images, self.n_keypoints, 2))
                h5file.create_dataset(
                    "annotations",
                    (self.n_images, self.n_keypoints, 2),
                    dtype=np.float64,
                    data=empty_array,
                    maxshape=(None, self.n_keypoints, 2),
                    chunks=True
                )
                for idx in range(self.n_images):
                    h5file["annotations"][idx] = self.skeleton.loc[:, ["x", "y"]].values

            if "annotated" not in list(h5file.keys()):
                empty_array = np.zeros((self.n_images, self.n_keypoints), dtype=bool)
                h5file.create_dataset(
                    "annotated",
                    (self.n_images, self.n_keypoints),
                    dtype=bool,
                    data=empty_array,
                    maxshape=(None, self.n_keypoints),
                    chunks=True
                )

            if "skeleton" not in list(h5file.keys()):
                skeleton = self.skeleton[["tree", "swap_index"]].values
                h5file.create_dataset(
                    "skeleton", skeleton.shape, dtype=np.int32, data=skeleton
                )

            # Unpack the images from the file
            self.image_idx = np.sum(np.all(h5file["annotated"].value, axis=1)) - 1
            if self.image_idx < 0:
                self.image_idx = 0
                
            self.image = h5file[self.dataset][self.image_idx]
            self._check_grayscale()
            
#             print(self.skeleton.loc[:, "annotated"])
#             print(h5file["annotated"][self.image_idx])
            self.skeleton.loc[:, ["x", "y"]] = h5file["annotations"][self.image_idx]
            self.skeleton.loc[:, "annotated"] = h5file["annotated"][self.image_idx]

    def _save(self):
        """ Saves an image.

        Automatically called when moving to new images or invoked manually
        using 'ctrl + s' keys.

        """
        with h5py.File(self.datapath) as h5file:

            h5file["annotations"][self.image_idx] = self.skeleton.loc[
                :, ["x", "y"]
            ].values
            h5file["annotated"][self.image_idx] = self.skeleton.loc[
                :, "annotated"
            ].values
            self.skeleton.loc[:, ["x", "y"]] = h5file["annotations"][self.image_idx]
            self.skeleton.loc[:, "annotated"] = h5file["annotated"][self.image_idx]

    def _load(self):
        """ Loads an image.

        This method is called in _move_image_idx when moving to different
        images. The image of specified image_idx will be loaded onto the GUI.

        """

        with h5py.File(self.datapath) as h5file:

            self.image = h5file[self.dataset][self.image_idx]
            self._check_grayscale()
            self.skeleton.loc[:, ["x", "y"]] = h5file["annotations"][self.image_idx]
            self.skeleton.loc[:, "annotated"] = h5file["annotated"][self.image_idx]

    def _new_annotation(self):
        """ Create new person annotation
        
        Adds new (image, empty_annotation) pair for persons completely 
        missed by Pose estimation model. Also, increase person_count for the image
        """
        try:
            if self.key is keys.N:
                with h5py.File(self.datapath) as hf:
                    new_image = self.image
#                     new_image = self.image.reshape(1, *self.image.shape)

                    # Add new image
                    hf[self.dataset].resize((hf[self.dataset].shape[0] + 1), axis = 0)

                    # hf[self.dataset] is np.ndarray
                    # last element will be null as its just created
#                     hf[self.dataset] = list(hf[self.dataset][:self.image_idx+1]) + list(new_image) + list(hf[self.dataset][self.image_idx+1:])
                    # move images after the current image by one place
                    hf[self.dataset][self.image_idx+2:] = hf[self.dataset][self.image_idx+1:-1]
                    
                    # add new image next to the current indexed image
                    hf[self.dataset][self.image_idx+1] = new_image
                       
                      
                    
#                     a = list(hf[self.dataset][:self.image_idx+1]) + list(new_image) + list(hf[self.dataset][self.image_idx+1:])
                    print(f'Annotator._new_annotation(): new image added {hf[self.dataset].shape}')
#                     print(f'Annotator._new_annotation(): new image added {len(a)}, {len(a[0])}, {a[0].shape}')
#                     print(f'Image index {self.image_idx}')
#                     print(np.array(hf[self.dataset][:self.image_idx+1]).shape)
#                     print(np.array(new_image).shape)
#                     print(np.array(hf[self.dataset][self.image_idx+1:]).shape)
                
                    # Add new empty annotation                       
#                     empty_annotation = np.zeros((1, *hf['annotations'].shape[1:]))
                    empty_annotation = hf['annotations'][self.image_idx]
                    hf["annotations"].resize((hf["annotations"].shape[0] + 1), axis = 0)
#                     hf["annotations"] = list(hf["annotations"][:self.image_idx+1]) + list(empty_annotation) + list(hf["annotations"][self.image_idx+1:])

                    # move anntns after the current annotation by one place
                    hf["annotations"][self.image_idx+2:] = hf["annotations"][self.image_idx+1:-1]
                    
                    # add new annotation next to the current indexed annotation
                    hf["annotations"][self.image_idx+1] = empty_annotation
                    
                    
                    print(f'Annotator._new_annoation(): new annotation added {hf["annotations"].shape}')
#                     b = list(hf["annotations"][:self.image_idx+1]) + list(empty_annotation) + list(hf["annotations"][self.image_idx+1:])
#                     print(f'Annotator._new_annoation(): new annotation added {len(b)}, {len(b[0])}, {b[0].shape}')

                    # Add annotated 
                    # Add new empty annotation                       
                    empty_annotated = np.zeros((1, self.n_keypoints), dtype=bool)
                    hf["annotated"].resize((hf["annotated"].shape[0] + 1), axis = 0)
                          
                    # move anntns after the current annotation by one place
                    hf["annotated"][self.image_idx+2:] = hf["annotated"][self.image_idx+1:-1]
                    
                    # add new annotation next to the current indexed annotation
                    hf["annotated"][self.image_idx+1] = empty_annotated
#                     hf["annotated"] = list(hf["annotated"][:self.image_idx+1]) + list(empty_annotated) + list(hf["annotated"][self.image_idx+1:-1])
                    print(f'Annotator._new_annoation(): new annotation added {hf["annotated"].shape}')
#                     b = list(hf["annotated"][:self.image_idx+1]) + list(empty_annotated) + list(hf["annotated"][self.image_idx+1:])
#                     print(f'Annotator._new_annoation(): new annotated added {len(b)}, {len(b[0])}, {b[0].shape}')

                    self.image_idx += 1
                    self.n_images += 1
                    self._load()
        except Exception as e:
            raise e
            print('Error in Annotator._new_annoation()')
            print(e)
             

            
        
        
    def _last_image(self):
        """ Checks if image index is on the last index.
    
        Helper method to check for the index of the last image in the h5 file.

        Returns
        -------
        bool
            Indicate if image_idx is the last index.

        """

        return self.image_idx == self.n_images - 1

    def _move_image_idx(self):
        """ Move to different image.
        
        Based on the key pressed, updates the image on the GUI.
        The scheme is as follows:
        ------------------------------------------------------------
             Keys             |   Action                           
        ------------------------------------------------------------
        >    <- , ->          |   Load previous or next image
        >    ,  ,  .          |   Jump 10 images backward or forward
        ------------------------------------------------------------

        Every time the user moves from the image, the annotations
        on the image is saved before loading the next image.

        """

        # <- (left arrow) key
        if self.key is keys.LEFTARROW:
            self._save()
            if self.image_idx == 0:
                self.image_idx = self.n_images - 1
            else:
                self.image_idx -= 1
            self._load()
        # -> (right arrow) key
        elif self.key is keys.RIGHTARROW:
            self._save()
            if self._last_image():
                self.image_idx = 0
            else:
                self.image_idx += 1
            self._load()

        # . (period) key
        elif self.key is keys.LESSTHAN:
            self._save()
            if self.image_idx - 10 < 0:
                self.image_idx = self.n_images + self.image_idx - 10
            else:
                self.image_idx -= 10
            self._load()
        # , (comma) key
        elif self.key is keys.GREATERTHAN:
            self._save()
            if self.image_idx + 10 > self.n_images - 1:
                self.image_idx = self.image_idx + 10 - self.n_images
            else:
                self.image_idx += 10
            self._load()

    def _data(self):
        """ Activates key bindings for annotated and save.
        
        Creates additional key bindings for the program.
        The bindings are as follows:
        
        ------------------------------------------------------------
             Keys             |   Action                           
        ------------------------------------------------------------
        >    Ctrl-R           |   Mark frame as unannotated
        >    Ctrl-F           |   Mark frame as annotated
        >    Ctrl-S           |   Save
        ------------------------------------------------------------

        """

        if self.key is keys.R:
            self.skeleton["annotated"] = False
        elif self.key is keys.F:
            self.skeleton["annotated"] = True
        elif self.key in [keys.Q, keys.ESC]:
            self._save()
            print("Saved")

    def _hotkeys(self):
        """ Activates all key bindings.
        
        Enables all the key functionalities described at the
        start of the file.

        """

        if self.key != keys.NONE:
            self._new_annotation()
            self._wasd()
            self._move_idx()
            self._move_image_idx()
            self._zoom()
            self._data()
            self._update_canvas()
