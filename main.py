from collections import defaultdict
from copy import deepcopy
import os
import numpy as np
from numpy import genfromtxt
import zipfile
import cv2
import math
import skimage.filter
import shutil

OUTPUT_DIR = '/tmp/1'
DATABASE_LOCATION = '/home/deathnik/src/my/magister/webcrdf-testbed/webcrdf-testbed/data/datadb.segmxr/'


def adjust_image(img, perc):
    im = img.astype(np.float)
    tbrd = math.floor(0.1 * np.min(im.shape))
    imc = im[tbrd:-tbrd, tbrd:-tbrd]
    q0, q1 = np.percentile(imc[:], [perc, 100.0 - perc])
    imm = np.max(im[:])
    im = 255. * (im - q0) / ((2.0 * perc * imm / 100.) + q1 - q0)
    im[im < 0] = 0
    im[im > 255] = 255
    return im


def make_masked_image(pts, img_path, msk):
    img = cv2.imread(img_path, 1)
    tmp = img[:, :, 2]
    tmp[msk > 0] = 255
    img[:, :, 2] = tmp
    if pts is not None:
        num_points = pts.shape[0]
        scale_size = np.min((img.shape[0], img.shape[1]))
        prad = int(5. * scale_size / 256.)
        for pp in xrange(0, num_points):
            cpp = pts[pp]
            cv2.circle(img, (int(cpp[0]), int(cpp[1])), prad + 2, (255, 255, 255), -1)  # cv2.cv.FILLED
            cv2.circle(img, (int(cpp[0]), int(cpp[1])), prad + 0, (0, 0, 255), -1)  # cv2.cv.FILLED
    return img


def make_img_on_mask(img_path, msk):
    tmp_img = cv2.imread(img_path, 0).astype(np.int32)
    tmp_img = tmp_img + 1
    tmp_img[tmp_img > 255] = 255
    tmp_img[msk == 0] = 0
    tmp_img = tmp_img.astype(np.uint8)
    return tmp_img


class Fix(object):
    elastix_command = "elastix -f {} -m {} -p {} -out {} -threads 4 >/dev/null"
    transformix_command = "transformix -in {} -tp {}/TransformParameters.0.txt -out {} >/dev/null"

    def __init__(self):
        pass

    @staticmethod
    def prepare_img(img_path):
        img = cv2.imread(img_path, 0).astype(np.float64)
        original_size = img.shape
        img = cv2.resize(img, (256, 256))
        return original_size, adjust_image(img, 1.0)

    @staticmethod
    def create_points(img, sum_points):
        points_number = 7
        points_xy = np.zeros((points_number, 2), np.float64)
        kxy = (img.shape[1] / 256., img.shape[0] / 256.)
        for pp in xrange(0, 7):
            pvalue = 100 + 20 * pp
            xy = np.where(sum_points == pvalue)
            if len(xy[0]) > 0:
                xym = np.mean(xy, 1)
                points_xy[pp, 0] = xym[1] * kxy[0]
                points_xy[pp, 1] = xym[0] * kxy[1]
            else:
                points_xy[pp, 0] = -1.0
                points_xy[pp, 1] = -1.0
        return points_xy

    def register_mask(self, img_path):
        original_size, img = self.prepare_img(img_path)

        images = [2]
        temp_out_mask = "%s/%s" % (OUTPUT_DIR, "result.bmp")
        temp_input = "%s/fix.png" % OUTPUT_DIR
        cv2.imwrite(temp_input, np.uint8(img))
        sum_points = None
        sum_correlation = 0.0
        max_correlation = -100
        for ii in images:
            # print "prcess %d : %s" % (ii, self.dataFnImg[ii])
            temp_database_img = os.path.join(DATABASE_LOCATION, '00{}.png'.format(ii))
            temp_database_mask = os.path.join(DATABASE_LOCATION, '00{}.bmp'.format(ii))
            temp_database_points = '%s_pts.png' % temp_database_img

            elastix_parameters = os.path.join(os.path.dirname(__file__), 'data/parameters_BSpline.txt')
            os.system(self.elastix_command.format(temp_input, temp_database_img, elastix_parameters, OUTPUT_DIR))
            os.system(self.transformix_command.format(temp_database_mask, OUTPUT_DIR, OUTPUT_DIR))

            # mask update
            print temp_out_mask
            temp_mask = cv2.imread(temp_out_mask, 0).astype(np.float) / 255.0
            temp_mask = skimage.filter.gaussian_filter(temp_mask, 0.5)
            if 'sum_mask' not in locals():
                sum_mask = temp_mask
            else:
                sum_mask += temp_mask

            # corr update
            temp_out_img = os.path.join(OUTPUT_DIR, 'result.0.bmp')
            temp_img = cv2.imread(temp_out_img, 0).astype(np.float)
            cur_correlation = np.corrcoef(img[20:-20].reshape(-1), temp_img[20:-20].reshape(-1))[0, 1]
            sum_correlation += cur_correlation

            # PTS
            os.system(self.transformix_command.format(temp_database_points, OUTPUT_DIR, OUTPUT_DIR))
            temp_points = cv2.imread(temp_out_mask, 0)
            if max_correlation < cur_correlation:
                sum_points = temp_points.copy()
                max_correlation = cur_correlation

        sum_correlation /= len(images)

        # pam pam
        ret = (sum_mask / len(images))
        ret = cv2.resize(ret, tuple(reversed(original_size)), interpolation=2)
        ret = (ret > 0.5)
        ret = 255 * np.uint8(ret)

        ### PTS-array (prepare)
        points_xy = self.create_points(ret, sum_points)

        ### PTS-array
        self.newMsk = ret
        self.pts = points_xy.copy()
        self.newImgMsk = make_masked_image(self.pts, img_path, self.newMsk)
        self.newImgOnMsk = make_img_on_mask(img_path, self.newMsk)
        return (ret, sum_correlation, sum_points, points_xy)


def load_database(db_size):
    items_seen = 0
    img_ids = []
    for i in xrange(2, db_size + 1):
        csv_path = os.path.join(DATABASE_LOCATION, '%03d.png_pts.csv' % i)
        try:
            my_data = genfromtxt(csv_path, delimiter=',')
        except IOError:
            continue
        # clean data

        # for _i in my_data:
        #    if _i[0] < 1 or _i [1] < 1:
        #        continue

        if 'avrg' not in locals():
            avrg = my_data
            data = my_data.flatten()
        else:
            avrg += my_data
            data = np.append(data, my_data.flatten())
        items_seen += 1
        img_ids.append(i)
    data = data.reshape(items_seen, 14)
    print data.shape
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(data)
    dists, points_indx = nbrs.kneighbors(my_data.flatten(), n_neighbors=2)
    avrg = (avrg / items_seen).flatten()
    print data
    transform = data - avrg
    return transform, nbrs, img_ids, avrg


def new_points(points, self_transform, other_transform, avrg):
    weights = []
    for i in xrange(0, len(points), 2):
        partial_weights = []
        for j in xrange(0, 14, 2):
            dst = math.sqrt(math.pow(points[i] - avrg[j], 2) + math.pow(points[i + 1] - avrg[j + 1], 2))
            partial_weights.append(dst)
        s = sum(partial_weights)
        weights.append([w / s for w in partial_weights])
    _new_points = deepcopy(points)
    transform = self_transform + other_transform
    for i in xrange(0, len(_new_points), 2):
        _new_points[i] += sum(transform[j] * weights[i / 2][j / 2] for j in xrange(0, 14, 2))
        _new_points[i + 1] += sum(transform[j + 1] * weights[i / 2][j / 2] for j in xrange(0, 14, 2))
    return _new_points


def draw_points_on_img(img_path, pts):
    img = cv2.imread(img_path, 1)

    if pts != None:
        numPts = pts.shape[0]
        scaleSiz = np.min((img.shape[0], img.shape[1]))
        prad = int(5. * scaleSiz / 256.)
        for pp in xrange(0, numPts):
            cpp = pts[pp]
            cv2.circle(img, (int(cpp[0]), int(cpp[1])), prad + 2, (255, 255, 255), -1)  # cv2.cv.FILLED
            cv2.circle(img, (int(cpp[0]), int(cpp[1])), prad + 0, (0, 0, 255), -1)  # cv2.cv.FILLED

    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ImageDB(object):
    def __init__(self):
        transform_matrix, nbrs_clf, img_ids, avrg = load_database(200)
        self.transform_matrix = transform_matrix
        self.nbrs_clf = nbrs_clf
        self.img_ids = img_ids
        self.avrg = avrg

    def do_magic(self, img_path, upper, lower):
        bound = np.array([upper[0], upper[1], upper[0], lower[1], lower[0], lower[1], lower[0], upper[1]])
        f = Fix()
        retMsk, retCorr, sumPts, ptsXY = f.register_mask(img_path)
        pts = f.pts.flatten()
        self_transform = self.avrg - pts
        _, indexes = self.nbrs_clf.kneighbors(pts, n_neighbors=5)
        indexes = indexes[0]
        draw_points_on_img(img_path, np.ndarray((4, 2), buffer=bound, dtype=float))
        for ind in indexes:
            pts = np.ndarray((4, 2), buffer=bound, dtype=float).flatten()
            p = new_points(pts, self_transform, self.transform_matrix[ind], self.avrg)
            p = p.reshape(4, 2)
            draw_points_on_img(os.path.join(DATABASE_LOCATION, '%03d.png' % self.img_ids[ind]), p)


def magic_method(img_path):
    transform_matrix, nbrs_clf, img_ids, avrg = load_database(200)
    f = Fix()
    retMsk, retCorr, sumPts, ptsXY = f.register_mask(img_path)
    pts = f.pts.flatten()
    self_transform = avrg - pts
    _, indexes = nbrs_clf.kneighbors(pts, n_neighbors=5)
    indexes = indexes[0]
    draw_points_on_img(img_path,
                       np.ndarray((4, 2), buffer=np.array([64.0, 64, 64, 128, 128, 128, 128, 64]), dtype=float))
    for ind in indexes:
        pts = np.ndarray((4, 2), buffer=np.array([64.0, 64, 64, 128, 128, 128, 128, 64]), dtype=float).flatten()
        p = new_points(pts, self_transform, transform_matrix[ind], avrg)
        p = p.reshape(4, 2)
        draw_points_on_img(os.path.join(DATABASE_LOCATION, '%03d.png' % img_ids[ind]), p)

    return

    pathImgMask = "%s_mask.png" % img_path
    pathImgMasked = "%s_masked.png" % img_path
    pathImgOnMask = "%s_onmask.png" % img_path
    pathPtsCSV = "%s_pts.csv" % img_path
    print f.pts
    if retCorr > 0.7:
        cv2.imwrite(pathImgMask, f.newMsk)
        cv2.imwrite(pathImgMasked, f.newImgMsk)
        cv2.imwrite(pathImgOnMask, f.newImgOnMsk)
        np.savetxt(pathPtsCSV, f.pts, delimiter=',', newline='\n', fmt='%0.1f')
    else:
        tmpNewImgMsk = cv2.imread(img_path, 1)  # cv2.IMREAD_COLOR)
        tmpImgOnMsk = np.zeros((tmpNewImgMsk.shape[0], tmpNewImgMsk.shape[1]), np.uint8)
        p00 = (0, 0)
        p01 = (0, tmpNewImgMsk.shape[0])
        p10 = (tmpNewImgMsk.shape[1], 0)
        p11 = (tmpNewImgMsk.shape[1], tmpNewImgMsk.shape[0])
        cv2.line(tmpNewImgMsk, p00, p11, (0, 0, 255), 4)
        cv2.line(tmpNewImgMsk, p01, p10, (0, 0, 255), 4)
        f.newMsk[:] = 0
        cv2.imwrite(pathImgMask, f.newMsk)
        cv2.imwrite(pathImgMasked, tmpNewImgMsk)
        cv2.imwrite(pathImgOnMask, tmpImgOnMsk)
        tmpPts = np.zeros((7, 2), np.float64)
        np.savetxt(tmpPts, delimiter=',', newline='\n')
        fnErr = "%s.err" % img_path
        f = open(fnErr, 'w')
        f.close()
    fzip = "%s.zip" % img_path
    zObj = zipfile.ZipFile(fzip, 'w')
    zipDir = '%s_dir' % os.path.basename(img_path)
    lstFimg = (img_path, pathImgMask, pathImgMasked, pathImgOnMask, pathPtsCSV)
    for ff in lstFimg:
        ffbn = os.path.basename(ff)
        zObj.write(ff, "%s/%s" % (zipDir, ffbn))
    print "retCorr = %s" % retCorr


def main():
    magic_method('/home/deathnik/src/my/magister/webcrdf-testbed/webcrdf-testbed/data/datadb.segmxr/001.png')


if __name__ == "__main__":
    main()