# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from beacon_aug.generator.operator_generator import Operator_generator
import beacon_aug.generator.custom as custom
import beacon_aug.generator.gan_based as gan_based

import json
import yaml
import glob
import os
from beacon_aug.generator.docs_generator import generate_doc

for_website = False


def class_definition(cls,  library=None, *args, **kwargs):
    return Operator_generator(cls.__name__, library=library, *args, **kwargs)


""" Standard operator wrapper
# dynamic class definition for all customize operators,
# s.t. the operators can be called as BA.{standard_operator }(library = "iaa")

"""


def doc_definition(op_name):
    return generate_doc(op_name)


js_fnames = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "./generator/standard/*.yaml"))
for js_f in js_fnames:
    with open(js_f) as json_file:
        standard_all = json.load(json_file) if ".json" in js_f else yaml.safe_load(json_file)
        if standard_all is not None:
            for op_name in standard_all:
                locals()[str(op_name)] = type(str(op_name),
                                              (object,),
                                              {"__new__": class_definition,
                                               "__doc__": doc_definition
                                               })
                if for_website:
                    locals()[str(op_name)].__doc__ = doc_definition(op_name)

""" Customize operator wrapper
# dynamic class definition for all customize operators,
# s.t. the operators can be called as BA.{customize_operator}(library = "custom")

"""

for op_name in custom.__all__ + gan_based.__all__:
    # only add the operator names not appeared in standard
    if op_name not in locals():
        locals()[str(op_name)] = type(str(op_name),
                                      (object,),
                                      {"__new__": class_definition,
                                       "__doc__": doc_definition
                                       })
        if for_website:
            locals()[str(op_name)].__doc__ = doc_definition(op_name)
