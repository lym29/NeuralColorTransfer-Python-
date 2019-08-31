from scipy import sparse
import numpy as np


class LocalColorTransferFilter:
    """
    Estimate linear coefficients in color transfer T(S) = a S + b
    Source and guide image should be in CIELAB color space
    """
    def __init__(self, s, g, sf, gf):
        self.s_height = s.shape[0]
        self.s_width = s.shape[1]
        self.channel_num = s.shape[2]
        self.pixel_num = s.shape[0] * s.shape[1]
        self.sf = sf
        self.gf = gf
        self.s_pixel = np.reshape(s, (self.pixel_num, self.channel_num))
        self.g_pixel = np.reshape(g, (self.pixel_num, self.channel_num))

    def construct_linsys_d(self, L):
        w = self.compute_weight_d(self.sf, self.gf, L)
        weight = np.repeat(w.reshape((-1, 1)), 3, axis=1)
        I1 = J1 = np.arange(self.pixel_num, dtype=int)
        V1 = weight * self.s_pixel * self.s_pixel

        I2 = I1
        J2 = self.pixel_num + J1
        V2 = weight * self.s_pixel

        I3 = self.pixel_num + I1
        J3 = J1
        V3 = weight * self.s_pixel

        I4 = J4 = self.pixel_num + I1
        V4 = weight * np.ones((self.pixel_num, self.channel_num))

        I = np.concatenate((I1, I2, I3, I4), axis=0)
        J = np.concatenate((J1, J2, J3, J4), axis=0)
        V = np.concatenate((V1, V2, V3, V4), axis=0)

        right_vec = np.concatenate((weight * self.s_pixel * self.g_pixel, self.g_pixel), axis=0)
        return I, J, V, right_vec

    def compute_weight_d(self, sf, gf, L):
        f_dim = sf.shape[2]
        s_feat = np.reshape(sf, (self.pixel_num, f_dim))
        g_feat = np.reshape(gf, (self.pixel_num, f_dim))
        matching_error = np.power(s_feat - g_feat, 2).sum(axis=-1)
        confidence = 1 - matching_error
        confidence[confidence < 1e-6] = 1e-6
        weight = pow(4, L-1) * confidence
        return weight

    def construct_linsys_l(self):
        return

    def solve(self, L):
        I_d, J_d, V_d, r_vec_d = self.construct_linsys_d(L)
        x = np.zeros((2 * self.pixel_num, self.channel_num))
        for i in range(self.channel_num):
            mat_d = sparse.csc_matrix((V_d[:, i], (I_d, J_d)))
            mat = mat_d
            r_vec = r_vec_d[:, i]
            #x[:, i] = sparse.linalg.spsolve(mat, r_vec)
            x[:, i], istop, itn, normr = sparse.linalg.lsmr(mat, r_vec)[:4]

        a = np.reshape(x[:self.pixel_num, :], (self.s_height, self.s_width, self.channel_num))
        b = np.reshape(x[self.pixel_num:, :], (self.s_height, self.s_width, self.channel_num))
        return a, b









