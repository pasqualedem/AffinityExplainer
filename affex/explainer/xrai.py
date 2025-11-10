# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of XRAI algorithm adapted for FSS.

Paper: https://arxiv.org/abs/1906.02825
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from saliency.core.xrai import (
    XRAI,
    XRAIParameters,
    XRAIOutput,
    _get_segments_felzenszwalb,
    _attr_aggregation_max,
    _get_diff_cnt,
    _get_diff_mask,
    _gain_density,
)

import numpy as np

_logger = logging.getLogger(__name__)


class FSSXRAI(XRAI):
    """XRAI class adapted for FSS."""

    def GetMaskWithDetails(
        self,
        x_value,
        call_model_function,
        call_model_args=None,
        baselines=None,
        segments=None,
        base_attribution=None,
        batch_size=1,
        extra_parameters=None,
    ):
        """Applies XRAI method on an input image and returns detailed information.


        Args:
            x_value: Input ndarray.
            call_model_function: A function that interfaces with a model to return
              specific data in a dictionary when given an input and other arguments.
              Expected function signature:
              - call_model_function(x_value_batch,
                                    call_model_args=None,
                                    expected_keys=None):
                x_value_batch - Input for the model, given as a batch (i.e.
                  dimension 0 is the batch dimension, dimensions 1 through n
                  represent a single input).
                call_model_args - Other arguments used to call and run the model.
                expected_keys - List of keys that are expected in the output. For
                  this method (XRAI), the expected keys are
                  INPUT_OUTPUT_GRADIENTS - Gradients of the output being
                    explained (the logit/softmax value) with respect to the input.
                    Shape should be the same shape as x_value_batch.
            call_model_args: The arguments that will be passed to the call model
               function, for every call of the model.
            baselines: a list of baselines to use for calculating
              Integrated Gradients attribution. Every baseline in
              the list should have the same dimensions as the
              input. If the value is not set then the algorithm
              will make the best effort to select default
              baselines. Defaults to None.
            segments: the list of precalculated image segments that should
              be passed to XRAI. Each element of the list is an
              [N,M] boolean array, where NxM are the image
              dimensions. Each elemeent on the list contains exactly the
              mask that corresponds to one segment. If the value is None,
              Felzenszwalb's segmentation algorithm will be applied.
              Defaults to None.
            base_attribution: an optional pre-calculated base attribution that XRAI
              should use. The shape of the parameter should match
              the shape of `x_value`. If the value is None, the
              method calculates Integrated Gradients attribution and
              uses it.
            batch_size: Maximum number of x inputs (steps along the integration
              path) that are passed to call_model_function as a batch.
            extra_parameters: an XRAIParameters object that specifies
              additional parameters for the XRAI saliency
              method. If it is None, an XRAIParameters object
              will be created with default parameters. See
              XRAIParameters for more details.

        Raises:
            ValueError: If algorithm type is unknown (not full or fast).
                        If the shape of `base_attribution` dosn't match the shape of
                          `x_value`.
                        If the shape of INPUT_OUTPUT_GRADIENTS doesn't match the
                          shape of x_value_batch.

        Returns:
            XRAIOutput: an object that contains the output of the XRAI algorithm.

        TODO(tolgab) Add output_selector functionality from XRAI API doc
        """
        if extra_parameters is None:
            extra_parameters = XRAIParameters(steps=25, algorithm="fast")

        # Check the shape of base_attribution.
        if base_attribution is not None:
            if not isinstance(base_attribution, np.ndarray):
                base_attribution = np.array(base_attribution)
            if base_attribution.shape != x_value.shape:
                raise ValueError(
                    "The base attribution shape should be the same as the shape of "
                    "`x_value`. Expected {}, got {}".format(
                        x_value.shape, base_attribution.shape
                    )
                )

        # Calculate IG attribution if not provided by the caller.
        if base_attribution is None:
            _logger.info("Computing IG...")
            x_baselines = self._make_baselines(x_value, baselines)

            attrs = self._get_integrated_gradients(
                x_value,
                call_model_function,
                call_model_args=call_model_args,
                baselines=x_baselines,
                steps=extra_parameters.steps,
                batch_size=batch_size,
            )
            # Merge attributions from different baselines.
            attr = np.mean(attrs, axis=0)
        else:
            x_baselines = None
            attrs = base_attribution
            attr = base_attribution

        # Merge attribution channels for XRAI input
        if len(attr.shape) > 2:
            attr = _attr_aggregation_max(attr, axis=1)

        _logger.info("Done with IG. Computing XRAI...")
        if segments is not None:
            segs = segments
        else:
            segs = [
                _get_segments_felzenszwalb(x_value[i].transpose(1, 2, 0))
                for i in range(1, x_value.shape[0])
            ]

        support_images = attr.shape[0] - 1
        support_attrs = attr[1 : support_images + 1]

        attr_maps = []
        attr_datas = []

        for i in range(support_images):
            seg = segs[i]
            attr = support_attrs[i]

            if extra_parameters.algorithm == "full":
                attr_map, attr_data = self._xrai(
                    attr=attr,
                    segs=seg,
                    area_perc_th=extra_parameters.area_threshold,
                    min_pixel_diff=extra_parameters.experimental_params[
                        "min_pixel_diff"
                    ],
                    gain_fun=_gain_density,
                    integer_segments=extra_parameters.flatten_xrai_segments,
                )
            elif extra_parameters.algorithm == "fast":
                attr_map, attr_data = self._xrai_fast(
                    attr=attr,
                    segs=seg,
                    min_pixel_diff=extra_parameters.experimental_params[
                        "min_pixel_diff"
                    ],
                    gain_fun=_gain_density,
                    integer_segments=extra_parameters.flatten_xrai_segments,
                )
            else:
                raise ValueError(
                    "Unknown algorithm type: {}".format(extra_parameters.algorithm)
                )
            attr_maps.append(attr_map)
            attr_datas.append(attr_data)

        # Concatenate the background attribution maps
        attr_map = np.stack(attr_maps)

        attr_data = attr_datas

        results = XRAIOutput(attr_map)
        results.baselines = x_baselines
        if extra_parameters.return_xrai_segments:
            results.segments = attr_data
        # TODO(tolgab) Enable return_baseline_predictions
        # if extra_parameters.return_baseline_predictions:
        #   baseline_predictions = []
        #   for baseline in x_baselines:
        #     baseline_predictions.append(self._predict(baseline))
        #   results.baseline_predictions = baseline_predictions
        if extra_parameters.return_ig_attributions:
            results.ig_attribution = attrs
        return results
