# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:47:56 on Sat, May 28, 2022
#
# Description: code generate for cuda-related dynamic libraries

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import optparse
from CppHeaderParser import CppHeader


class CodeGenerate():
    def __init__(self, type_, file_, output_):
        self.type = type_
        self.file = file_
        self.output = output_

        self.func_list = []

        self.hook_file = self.output + "/" + self.type + "_hook.cpp"
        self.hook_list = []
        self.hook_include = """
// auto generate $hook_num$ apis

#include "$type$_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"
"""
        self.hook_template = """
HOOK_C_API HOOK_DECL_EXPORT $ret$ $func_name$($func_param$) {
    HOOK_TRACE_PROFILE("$func_name$");
    using func_ptr = $ret$ (*)($param_type$);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_$type$_SYMBOL("$func_name$"));
    HOOK_CHECK(func_entry);
    return func_entry($param_name$);
}
"""

    def parsę_header(self):
        self.header = CppHeader(self.file)
        print(
            "{} total func num: {}".format(
                self.type, len(
                    self.header.functions)))

    def generate_func(self):
        for func in self.header.functions:
            func_name = func["name"]
            if func_name in self.func_list:
                continue
            else:
                self.func_list.append(func_name)

            ret = func["rtnType"].replace(
                "CUDAAPI", "").replace(
                "__CUDA_DEPRECATED", "").replace(
                "DECLDIR", "").replace(
                "CUDARTAPI_CDECL", "").replace(
                "CUDARTAPI", "").replace(
                "__host__", "").replace(
                "__cudart_builtin__", "").replace(
                "CUDNNWINAPI", "").replace(
                "CUBLASWINAPI", "").replace(
                "CUBLASAPI", "").replace(
                "CUFFTAPI", "").replace(
                "NVTX_DECLSPEC", "").replace(
                "NVTX_API", "").replace(
                "CURANDAPI", "").replace(
                "CUSPARSEAPI", "").replace(
                "CUSOLVERAPI", "").replace(
                "NVJPEGAPI", "").strip(' ')

            func_param = ""
            param_type = ""
            param_name = ""
            for param in func["parameters"]:
                if len(func_param) > 0:
                    func_param += ", "
                    param_type += ", "
                    param_name += ", "
                if param["array"] == 1:
                    param["type"] += "*"
                func_param += (param["type"] + " " + param["name"])
                param_type += param["type"]
                param_name += param["name"]

            hook_func = self.hook_template
            self.hook_list.append(
                hook_func.replace(
                    "$ret$",
                    ret).replace(
                    "$func_name$",
                    func_name).replace(
                    "$func_param$",
                    func_param).replace(
                    "$param_type$",
                    param_type).replace(
                        "$param_name$",
                        param_name).replace(
                            "$type$",
                    self.type.upper()))
        print("{} valid func num: {}".format(self.type, len(self.func_list)))

    def save_output(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        with open(self.hook_file, 'w') as fh:
            hook_include = self.hook_include.replace("$hook_num$", str(
                len(self.hook_list))).replace("$type$", self.type)
            fh.write(hook_include)
            for hook in self.hook_list:
                fh.write(hook)


def main():
    usage = "python3 code_generate.py -t/--type cuda -f/--file include/cuda.h -o/--output output"
    parser = optparse.OptionParser(usage)
    parser.add_option(
        '-t',
        '--type',
        dest='type',
        type='string',
        help='header type',
        default='cuda')
    parser.add_option(
        '-f',
        '--file',
        dest='file',
        type='string',
        help='header file',
        default='include/cuda.h')
    parser.add_option(
        '-o',
        '--output',
        dest='output',
        type='string',
        help='output path',
        default='output')

    options, args = parser.parse_args()
    type_ = options.type
    file_ = options.file
    output_ = options.output

    code_gen = CodeGenerate(type_, file_, output_)
    code_gen.parsę_header()
    code_gen.generate_func()
    code_gen.save_output()


if __name__ == '__main__':
    main()
