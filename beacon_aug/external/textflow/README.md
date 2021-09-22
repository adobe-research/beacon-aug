Complete version of Textflow (Text render, perturbation, and synthesis with natural images.
) please check: https://git.corp.adobe.com/zzhang/TextFlow/tree/master/text.

``Beacon_aug`` only adopts the ``text_synthesis`` part


## TextFlow.text_synthesis()
Synthesize image with text on it.

#### text_synthesis(img, text_list, font_list, region_list=None, effect_list=None, is_top_left_origin=True, ensure_num_regions=False, is_keep_ratio=True)
##### Inputs:
* img - str or array
    * str - path to an image
    * array - RGB image array
* text_list - a list of text in str
* font_list - a list of font file path in str
* region_list - None or a list of bounding quadrilaterals, None means automatically select suitable regions for placing text
* effect_list - None or a list dict, None means no effects on text. A dict contains all or part of the following items: 
    * 'layer_text': a list of 3 elements, [bool, RGB 0 ~ 255, float 0 ~ 1] indicates [whether feather, text color, opacity]
    * 'layer_border': a list of 2 elements, [bool, RGB 0 ~ 255] indicates [whether random color, border color]. If border color is not None, the first element is ignored.
    * 'layer_shadow': a list of 3 elements, [float 0 ~ 2pi, int >=0, float 0 ~ 1] indicates [theta, shift, opacity]
    * 'layer_background': RGB color, e.g., (100, 100, 100)
    * 'text_size': None or int, text height in pixel. None means setting the height automatically according to image region size
    * 'text_interval': float from 0.5 to 2.0, shrink (<1) or enlarge (>1) interval so that final_text_length / original_text_length = interval
    * 'mix_bg': bool, whether mix text to background image using poisson image editing [Perez et al.] (http://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
* is_top_left_origin - bool, whether keep the first point of a quadrilateral to be the top-left corner in x-y coord
* ensure_num_regions - bool, whether ensure the number of text placed on the image. Sometimes, there are not enough suitable regions to place all text, so some text would be omitted intentionally
* is_keep_ratio - bool, whether keep the spatial ratio of rendered text. If False, the text might show strong perspective transformation 

##### Return:
* dict:
    * 'img': RGB image array
    * 'txt': a list of M str, M = number of text placed on the image
    * 'bbx': Mx4x2 array, M text bounding quadrilaterals, each of which has 4 points in image (x-y) coord in the order of top-left, top-right, bottom-right, and bottom-left
    * 'cbx': MxNx4x2 array, N = number of characters, MxN character bounding quadrilaterals, each of which is in the same format as bbx
    * 'bln': Mx2x2 array, baselines of the M text, each baseline has two points indicating a line in 2D space

###### Synthesize image with text
```python
import TextFlow
import numpy

img_dir = 'TextFlow/text/data/image/input.png'
text_list = ['Text', 'Flow', 'dlg']
font_list = ['TextFlow/text/data/font/Vera.ttf', 'TextFlow/text/data/font/VeraMono.ttf']
effect_list = []
for _ in range(len(text_list)):
    params = {
        'layer_text': [numpy.random.choice(2), 'rand', numpy.random.uniform(.5, 1)],  # [random feather, random text color, random opacity]
        'layer_border': [True, None],  # [random border color, no specific color]
        'layer_shadow': [None, None, None],  # [random theta, random shift, random opacity]
        'layer_background': None,  # keep original background color
        'text_size': None,  # automatic size
        'text_interval': numpy.random.uniform(.8, 1.4),  # random interval
        'mix_bg': numpy.random.choice(2)  # randomly decision on mixing text to background
    }
    effect_list.append(params)

data = TextFlow.text_synthesis(
    img=img_dir,
    text_list=text_list,
    font_list=font_list,
    region_list=None,  # automatically select suitable text regions on the image
    effect_list=effect_list,
    is_top_left_origin=True,
    ensure_num_regions=False,
    is_keep_ratio=True,
)

img = data['img']
txt = data['txt']
bbx = data['bbx']
cbx = data['cbx']
bln = data['bln']
```
The synthesize image with text and annotation are shown in the following.

<img src="./results_normal/text_synthesis.png" width="50%"><img src="./results_annotation/text_synthesis.png" width="50%">


<a name="text_generator">

## TextFlow.text_generator() 
Create a generator of text or image with text.

#### text_generator(image_dir, text_dir, font_dir, is_perturb=True, interval=(.8, 1.2), height=(80, 120), k=(5, 10), img_size=None)
##### Inputs:
* image_dir - None or str, dir to image files, i.e., `.png` and `.jpg`. If None, text rendering mode
* text_dir - str, dir to text files, i.e., `.txt`
* font_dir - str, dir to font files, i.e., `.ttf`, `.otf`
* is_perturb - bool, whether randomly perturb the text. When image_dir is not None, is_perturb is ignored.
* interval - a range of text interval, text interval could be randomly set within this range
* height - a range of text height, text height in pixel could be randomly set within this range
* k - a range of number of text placed on a image, the number of text on a image could be randomly set within this range
* img_size: (h, w) if not None

##### Return
* a generator of dict, a dict contains:
    * 'img': RGB image array
    * 'txt': a list of M str, M = number of text placed on the image
    * 'bbx': Mx4x2 array, M text bounding quadrilaterals, each of which has 4 points in image (x-y) coord in the order of top-left, top-right, bottom-right, and bottom-left
    * 'cbx': MxNx4x2 array, N = number of characters, MxN character bounding quadrilaterals, each of which is in the same format as bbx
    * 'bln': Mx2x2 array, baselines of the M text, each baseline has two points indicating a line in 2D space

Note: `TextFlow.text_synthesis()` is called inside, where some parameters are set as constant, 
i.e., `is_top_left_origin=True`, `ensure_num_regions=Fale`, `is_keep_ratio=True`, `mix_bg=False`. 
These fixed parameters would ensure images with visible text and reasonable outline.  

###### Create generator of text or image with text
```python
import TextFlow

image_dir = 'TextFlow/text/data/image'
text_dir = 'TextFlow/text/data/corpus'
font_dir = 'TextFlow/text/data/font'

# generator of clean text
generator_txt_clean = TextFlow.text_generator(
    image_dir=None,
    text_dir=text_dir,
    font_dir=font_dir,
    is_perturb=False
)

# generator of perturbed text
generator_txt_perturb = TextFlow.text_generator(
    image_dir=None,
    text_dir=text_dir,
    font_dir=font_dir,
    is_perturb=True
)

# generator of image with text
generator_txt_img = TextFlow.text_generator(
    image_dir=image_dir,
    text_dir=text_dir,
    font_dir=font_dir,
)

# use a generator
for data in generator_txt_img:
    img = data['img']
    txt = data['txt']
    bbx = data['bbx']
    cbx = data['cbx']
    bln = data['bln']

    # save data as a synthetic dataset
    # or provide online training data
```

Some examples from the generator of image with text are shown in the following.
<img src="./results_normal/text_generator_00000.png" width="50%"><img src="./results_annotation/text_generator_00000.png" width="50%">

<img src="./results_normal/text_generator_00001.png" width="50%"><img src="./results_annotation/text_generator_00001.png" width="50%">

<img src="./results_normal/text_generator_00002.png" width="50%"><img src="./results_annotation/text_generator_00002.png" width="50%">

<img src="./results_normal/text_generator_00003.png" width="50%"><img src="./results_annotation/text_generator_00003.png" width="50%">

<img src="./results_normal/text_generator_00004.png" width="50%"><img src="./results_annotation/text_generator_00004.png" width="50%">


<a name="font_render">

## TextFlow.font_render()
Render font recognizable text image.

#### font_render(text, font, bg_img=None, bg=0, interval=1, height=80, params='rand')
##### Inputs:
* text - str, text to be rendered
* font - str, path to a font file
* bg_img - str or array, either path to the background image or an image array already loaded to memory
* bg - 0 or 1, background color if bg_img is None, 0-black, 1-white. The foreground/text color is opposite to background color
* interval: float from 0.5 to 2.0, shrink (<1) or enlarge (>1) interval so that final_text_length / original_text_length = interval
* height - int, height of rendered text image in pixel. The height of rendered text may be different because height in point is used in text rendering, and the transformation between pixel and point depends on screen resolution.  
* params - None, 'rand', dict, or list
    * None - render clean text
    * 'rand' - render text with random perturbation
    * dict - parameters of perturbation on text
        * 'PT_TRANSFORM': 4x2 array, [(top-left corner movement along x and y axis), (t-r ...), (b-r ...), (b-l ...)], value 0 ~ 1, the movements are ratio
        * 'PT_ROTATE': 1x1 array, [angle], value 0 ~ 1, projected to 0 ~ 360 in degree
        * 'PT_COLOR': 2x3 array, [foreground_RGB, background_RGB], value 0 ~ 1
        * 'PT_NOISE': 2x1 array, [noise_type, 1/scale], value 0 ~ 1, noise_type is projected to normal, uniform, or salt&pepper
        * 'PT_SHADOW': 5x1 array, [min_illumination, max_illumination, center_x, center_y, exposure]
    * list - a list of perturbation types list in dict above, e.g., ['PT_TRANSFORM', 'PT_NOISE'] 

##### Return:
* dict:
    * 'img': array, text image in gray scale if PT_COLOR is not set, otherwise return RGB image
    * 'txt': str, ground truth text
    * 'bbx': 4x2 array, text bounding quadrilateral, 4 points in image (x-y) coord in the order of top-left, top-right, bottom-right, and bottom-left
    * 'cbx': Nx4x2 array, N = number_of_characters, character bounding quadrilaterals, each of which is in the same format as bbx
    * 'bln': 2x2 array, baseline of the text, two points indicate a line in 2D space

###### Render font recognizable perturbed text
```python
import TextFlow
import numpy

text = 'TextFlow@DLG'
font = 'TextFlow/text/data/font/Vera.ttf'
bg_img='TextFlow/text/data/image/input.png'

# random interval/kerning, range [0.6, 1.6]
interval = numpy.random.rand() + .6

# specify perturbation types
params = ['PT_TRANSFORM', 'PT_TEXTURE', 'PT_NOISE', 'PT_COLOR']  # randomly apply specific perturbations
# params = 'rand'  # randomly apply all types of perturbations, which is equivalent to 
# params = ['PT_TRANSFORM', 'PT_TEXTURE', 'PT_NOISE', 'PT_ROTATE', 'PT_COLOR', 'PT_SHADOW']

data = TextFlow.font_render(text, font, bg_img=bg_img, bg=0, interval=interval, height=50, params=params)

img = data['img']
txt = data['txt']
bbx = data['bbx']
cbx = data['cbx']
bln = data['bln']
```
The rendered font recognizable text and annotation are shown in the following.

<img src="./results_normal/font_perturb.png" width="50%"><img src="./results_annotation/font_perturb.png" width="50%">

