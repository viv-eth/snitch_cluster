#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

import sys

from snitch.util.sim.data_utils import DataGen, format_struct_definition
from snitch.dnn import mha, mlp


class EncoderDataGen(DataGen):

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        mha_gen = mha.MHADataGen()
        mlp_gen = mlp.MLPDataGen()

        header += [mha_gen.emit_header(**kwargs['mha'])]
        header += [mlp_gen.emit_header(**kwargs['mlp'])]

        encoder_args = {
            'mha_args': '&mha_args',
            'mlp_args': '&mlp_args'
        }
        header += [format_struct_definition('encoder_args_t', 'encoder_args', encoder_args)]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    sys.exit(EncoderDataGen().main())
