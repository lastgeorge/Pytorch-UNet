#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def load(file, event, tags):
  data = h5py.File(file, 'r')
  frames = []
  for tag in tags:
    f = data.get('/%d/%s'%(event, tag))
    if f is None:
      return None
    frames.append(np.array(f))
  img = np.stack(frames, axis = 2)
  img = np.transpose(img, axes=[1, 0, 2])
  return img

def rebin(a, shape):
  sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
  if len(a.shape) == 3:
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1],a.shape[2]
  return a.reshape(sh).mean(3).mean(1)

def plot_img(img):
  for ich in range(img.shape[2]) :
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    a.set_title('CH{}'.format(ich))
    frame_ma = np.ma.array(np.transpose(img[:,:,ich], axes=[1, 0]))
    # plt.imshow(np.ma.masked_where(frame_ma<=0,frame_ma), cmap="bwr_r", origin='lower')
    plt.imshow(frame_ma, cmap="bwr", origin='lower')
    plt.clim(-1,1)
    # plt.colorbar()
    plt.grid()
  plt.show()

def plot_mask(mask):
  plt.figure()
  # plt.gca().set_title('Mask')
  plt.imshow(np.transpose(mask, axes=[1, 0])
  , cmap="bwr"
  , origin='lower'
  # , aspect='auto'
  )
  # print("Mask non-zero",np.count_nonzero(mask))
  # plt.colorbar()
  plt.clim(-1,1)
  plt.grid()
  plt.show()

def get_hwc_img(file, event, tags, scale, crop0, crop1, norm):
  """From a list of tuples, returns the correct cropped img"""
  im = load(file, event, tags)
  if im is None:
    return None
  im = rebin(im, [im.shape[0]//scale[0],im.shape[1]//scale[1]])/norm
  im = im[crop0[0]:crop0[1], crop1[0]:crop1[1], :]
  return im

def get_hwc_imgs(file, ids, tags, scale, crop0, crop1, norm):
  """From a list of tuples, returns the correct cropped img"""
  for id in ids:
    im = get_hwc_img(file[id[0]], id[1], tags, scale, crop0, crop1, norm)
    if im is None:
      print(f'warn: {file[id[0]]} {id[1]} {tags} is None!')
      continue
    yield im

def get_chw_imgs(file, ids, tags, scale, crop0, crop1, norm):
  """From a list of tuples, returns the correct cropped img"""
  for id in ids:
    im = get_hwc_img(file[id[0]], id[1], tags, scale, crop0, crop1, norm)
    if im is None:
      print(f'warn: {file[id[0]]} {id[1]} {tags} is None!')
      continue
    im = np.transpose(im, axes=[2, 0, 1])
    yield im

def get_masks(file, ids, tags, scale, crop0, crop1, threshold,
              padding=0, min_run=1,
              padding_side='both',   # 'both' | 'left' | 'right'
              avoid_merge=False,     # activate min_gap
              min_gap=1):
  """From a list of tuples, returns the correct cropped mask with optional padding on time axis."""
  if isinstance(file, (list, tuple)):
    files = file
  else:
    files = [file]

  def _find_runs(row):
    """row (W,) ∈ {0,1}. return starts, ends (exclusive)."""
    padded = np.pad(row, (1, 1), mode='constant', constant_values=0)
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return starts, ends  # row[starts[i]:ends[i]] == 1

  for id in ids:
    im = load(files[id[0]], id[1], tags)
    if im is None:
      print(f'warn: {files[id[0]]} {id[1]} {tags} is None!')
      continue

    if im.ndim == 3 and im.shape[2] == 1:
      im = im.reshape(im.shape[0], im.shape[1])
    elif im.ndim == 2:
      pass
    else:
      im = im[..., 0]

    # rebin & crop
    im = rebin(im, [im.shape[0]//scale[0], im.shape[1]//scale[1]])
    im = im[crop0[0]:crop0[1], crop1[0]:crop1[1]]

    # threshold
    im = (im > threshold).astype(np.float32)

    # padding
    if padding > 0 and min_run > 0:
      H, W = im.shape
      for h in range(H):
        row = im[h, :].copy()  # (W,)
        starts, ends = _find_runs(row)

        intervals = []  # [lo, hi, s, e, is_fixed]
        for s, e in zip(starts, ends):
          length = e - s
          if length >= min_run:
            pad_left  = padding if padding_side in ('left', 'both') else 0
            pad_right = padding if padding_side in ('right', 'both') else 0
            lo = max(0, s - pad_left)
            hi = min(W, e + pad_right)
            lo = min(lo, s)
            hi = max(hi, e)
            intervals.append([lo, hi, s, e, False])  # flex
          else:
            intervals.append([s, e, s, e, True])     # fixed

        if not intervals:
          im[h, :] = row
          continue

        if avoid_merge and len(intervals) > 1:
          intervals.sort(key=lambda x: x[0])

          for i in range(1, len(intervals)):
            prev_lo, prev_hi, prev_s, prev_e, prev_fixed = intervals[i-1]
            cur_lo,  cur_hi,  cur_s,  cur_e,  cur_fixed  = intervals[i]

            desired_lo = prev_hi + min_gap

            if cur_lo < desired_lo:
              deficit = desired_lo - cur_lo  # >= 1

              can_shrink_cur  = 0 if cur_fixed  else max(0, cur_s  - cur_lo)
              can_shrink_prev = 0 if prev_fixed else max(0, prev_hi - prev_e)

              half = deficit // 2
              d_cur  = min(can_shrink_cur,  half + (deficit % 2))
              d_prev = min(can_shrink_prev, deficit - d_cur)

              shortfall = deficit - (d_cur + d_prev)
              if shortfall > 0:
                add = min(shortfall, max(0, can_shrink_cur - d_cur))
                d_cur  += add
                shortfall -= add
              if shortfall > 0:
                add = min(shortfall, max(0, can_shrink_prev - d_prev))
                d_prev += add
                shortfall -= add

              if d_cur > 0:
                intervals[i][0] = cur_lo + d_cur
                cur_lo += d_cur
              if d_prev > 0:
                intervals[i-1][1] = prev_hi - d_prev
                prev_hi -= d_prev


        new_row = np.zeros_like(row)
        for lo, hi, s, e, _fixed in intervals:
          lo = int(max(0, lo)); hi = int(min(W, hi))
          if hi > lo:
            new_row[lo:hi] = 1.0
        for s, e in zip(starts, ends):
          new_row[s:e] = 1.0
        im[h, :] = new_row

    yield im

