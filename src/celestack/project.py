# # TODO: dissolve this file into the `stack` module

# import numpy as np
# import plotly.graph_objects as go

# from celestack.frame import Frame, Mask
# from celestack.segment import SegmentBox


# def segment_sky(mask: Mask, n_segments: int = 50) -> list[SegmentBox]:
#     """
#     Segment the unmasked portion of the light frame (the sky) into N rectangular
#     segments, with roughly the same number of sky pixels in each segment.

#     The segments are defined by their bounding boxes.

#     The segments are returned in a list of SegmentBox objects, in a continuous order,
#     meaning that each two segments next to each other in the list are also negboring
#     segments in the image.
#     More specifically, the segments are packed in the list column by column in a
#     meandering fashion, with the first segment being the top-left corner of the image.

#     Args:
#         mask: The mask to use for segmentation. This should be a Mask object.
#         n_segments: The number of segments to create. Defaults to 50.
#     """
#     segment_boxes = []

#     # First, figure out how many pixels we have in the sky:
#     ma = mask.mask_array  # This is a boolean array with False for sky pixels
#     h, w = mask.shape

#     # Lets define the xmin, xmax of the bounding box of the sky:
#     xmin, xmax = np.where(np.sum(ma, axis=0) < h)[0][[0, -1]]

#     # Let's define the nominal segment size (the segments will be nominally square):
#     n_px = np.sum(~ma)  # Total number of sky pixels
#     size_nom = int(np.sqrt(n_px // n_segments))

#     # The segments will be defined in columns - let's define the width of the column as
#     # how many times the nominal size fits between the xmin and xmax:
#     n_cols = int(round((xmax - xmin) / size_nom))
#     width_nom = (xmax - xmin) // n_cols

#     x1 = xmin
#     for i in range(n_cols):
#         x2 = x1 + width_nom if i < n_cols - 1 else xmax

#         # Now we need to find the ymin and ymax of the bounding box of the sky in the
#         # column:
#         ymin, ymax = np.where(np.sum(ma[:, x1:x2], axis=1) < (x2 - x1))[0][[0, -1]]

#         n_px_col = np.sum(~ma[ymin:ymax, x1:x2])  # Number of sky pixels in the column
#         n_rows = int(round(n_px_col / width_nom**2))
#         n_px_seg = n_px_col // n_rows  # Nominal number of pixels in the segment

#         j = 1
#         y1 = ymin
#         y2 = ymin
#         segment_boxes_col = []
#         while True:
#             # We'll increment y2 until we have enough pixels in the segment:
#             y2 += 1

#             # Catch the end of the column:
#             if j == n_rows:
#                 segment_boxes_col.append(SegmentBox(x1=x1, y1=y1, x2=x2, y2=ymax))
#                 if i % 2 == 1:
#                     segment_boxes_col.reverse()
#                 segment_boxes.extend(segment_boxes_col)
#                 segment_boxes_col = []
#                 break

#             # Check if we have enough pixels in the segment:
#             if np.sum(~ma[ymin:y2, x1:x2]) >= j * n_px_seg:
#                 segment_boxes_col.append(SegmentBox(x1=x1, y1=y1, x2=x2, y2=y2))
#                 y1 = y2
#                 j += 1

#         x1 = x2

#     return segment_boxes


# def plot_segments(frame: Frame, segment_boxes: list[SegmentBox]) -> go.Figure:
#     """
#     Plot the segment boxes on the frame.

#     Args:
#         frame: The frame to plot the segments on. This can be any subclass of the
#             Frame class, such as LightFrame or Mask.
#         segment_boxes: The list of segments to plot. This should be a list of SegmentBox
#             objects.

#     Returns:
#         A Plotly Figure object with the segments plotted on the frame.
#     """
#     # Plot the compressed image belonging to the frame:
#     fig = frame.plot()

#     # Add the segment boxes to the plot:
#     x, y = [], []
#     for box in segment_boxes:
#         x.extend([box.x1, box.x2, box.x2, box.x1, box.x1, np.nan])
#         y.extend([box.y1, box.y1, box.y2, box.y2, box.y1, np.nan])
#     fig.add_trace(
#         go.Scatter(
#             x=x,
#             y=y,
#             mode="lines",
#             line=dict(color="blue", width=0.5),
#             name="Segment boxes",
#         )
#     )
#     fig.update_layout(showlegend=False)

#     return fig
