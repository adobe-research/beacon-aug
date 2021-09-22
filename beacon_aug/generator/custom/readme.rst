
Guidance for adding  customized operator function
=================================================

beacon_aug will automatically mode this operator to augmentation operator


* Name: Class name has to be the same with python file name

  * if classname exist in yaml file in `\ ``standard`` <beacon_aug/generator/standard/>`_\ :

    * append a "custom" library function to the existing beacon_aug class

  * else:

    * create a beacon_aug class with the same name that only support the "custom" library

see example in `KeepSizeCrop.py <beacon_aug/generator/custom/KeepSizeCrop.py>`_

Test the operator function itself:    

.. code-block:: python

   from beacon_aug.generator.custom.KeepSizeCrop import KeepSizeCrop
   op = KeepSizeCrop( scale=(0.08, 1.0))
   img_transformed = op(img )

Test the augmentation wrapper:   

.. code-block:: python

   import beacon_aug as BA
   aug = BA.KeepSizeCrop(p=1, height=64,width=64, library="custom")
   image_auged = aug(image=image)["image"]
