class ObjectMotionCounterfactual:

    def __init__(self, ):
        pass

    def run_forward(self, image, start_points, end_points, K):
        '''
        :param image: [H, W, 3] in [0, 1] range
        :param start_points: [N, K, 2] a list of K points for each mask, K >=3
        :param end_points: [N, K, 2] a list of K points for each mask, K >=3
        :return:
            counterfactual_image: [H, W, 3]
        '''

    def run_forward_autoregressive(self, image, start_points, end_points, K):
        '''
        :param image: [H, W, 3] in [0, 1] range
        :param start_points: [N, K, 2] a list of K points for each mask, K >=3
        :param end_points: [N, K, 2] a list of K points for each mask, K >=3
        :return:
            counterfactual_images: [[H, W, 3]] list of images
        '''
