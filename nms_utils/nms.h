#include <vector>
#include <iostream>
#include <algorithm> 
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <assert.h>     /* assert */
#include <opencv2/opencv.hpp>

#ifndef OPEN_PTRACK_NMS_UTILS_NMS_
#define OPEN_PTRACK_NMS_UTILS_NMS_

namespace open_ptrack
{
  namespace nms_utils
  {
    class MatF{
    public:
        float* m_data;
        int m_rows, m_cols, m_channels;

    public:
        MatF(int cols, int rows, int channels){
            m_rows = rows;
            m_cols = cols;
            m_channels = channels;
            int size = channels * rows * cols * sizeof(float);
            m_data = (float*)malloc(size);
            memset((void *)m_data, 0, size);
        }
        // this during free tvmfree while a pointer causes some weird probs
        //~MatF(){
        //    if(m_data) free(m_data);
        //}

        float *at(int channel, int row, int col){
            assert(m_data != NULL);
            assert(row < m_rows);
            assert(col < m_cols);
            assert(channel < m_channels);

            return m_data + (channel * m_rows * m_cols) + row * m_cols + col;
        }

        int getRows() {return m_rows;}
        int getCols() {return m_cols;}
        int getChannels() {return m_channels;}
    };


    class sortable_result {
    public:
        sortable_result() {
        }

        ~sortable_result() {
        }

        bool operator<(const sortable_result &t) const {
            return probs < t.probs;
        }

        bool operator>(const sortable_result &t) const {
            return probs > t.probs;
        }

        float& operator[](int i) {
            assert(0 <= i && i <= 4);

            if (i == 0) 
                return xmin;
            if (i == 1) 
                return ymin;
            if (i == 2) 
                return xmax;
            if (i == 3) 
                return ymax;
            return xmin;
        }

        float operator[](int i) const {
            assert(0 <= i && i <= 4);

            if (i == 0) 
                return xmin;
            if (i == 1) 
                return ymin;
            if (i == 2) 
                return xmax;
            if (i == 3) 
                return ymax;
            else{
                return 0;
            }
        }

        int index;
        int cls;
        float probs;
        float xmin;
        float ymin;
        float xmax;
        float ymax;

        float area(){
            return (xmax - xmin) * (ymax - ymin);
        }
    };


    class sortable_mask_result {
    public:
        sortable_mask_result() {
        }

        ~sortable_mask_result() {
        }

        bool operator<(const sortable_mask_result &t) const {
            return probs < t.probs;
        }

        bool operator>(const sortable_mask_result &t) const {
            return probs > t.probs;
        }

        float& operator[](int i) {
            assert(0 <= i && i <= 4);

            if (i == 0) 
                return xmin;
            if (i == 1) 
                return ymin;
            if (i == 2) 
                return xmax;
            if (i == 3) 
                return ymax;
            return xmin;
        }

        float operator[](int i) const {
            assert(0 <= i && i <= 4);

            if (i == 0) 
                return xmin;
            if (i == 1) 
                return ymin;
            if (i == 2) 
                return xmax;
            if (i == 3) 
                return ymax;
            else{
                return 0;
            }
        }

        int index;
        int cls;
        float probs;
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        int global_idx;

        float area(){
            return (xmax - xmin) * (ymax - ymin);
        }
    };

    void tvm_nms_cpu(std::vector<sortable_result>& boxes, MatF tvm_output, float cls_threshold, float nms_threshold, std::vector<sortable_result>& filterOutBoxes);

    /// methods for opencv based nms

    //template<typename _Tp> static inline
    //double jaccardDistance(const cv::Rect_<_Tp>& a, const cv::Rect_<_Tp>& b);

    template <typename T>
    static inline float rectOverlap(const T& a, const T& b);

    template <typename T>
    static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                            const std::pair<float, T>& pair2);

    inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                        std::vector<std::pair<float, int> >& score_index_vec);

    template <typename BoxType>
    inline void NMSFast_(const std::vector<BoxType>& bboxes,
        const std::vector<float>& scores, const float score_threshold,
        const float nms_threshold, const float eta, const int top_k,
        std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&));


    void NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores,
                            const float score_threshold, const float nms_threshold,
                            std::vector<int>& indices);

    void opencv_nms(MatF tvm_output, float cls_threshold, float nms_threshold, std::vector<sortable_result>& boxes);

    /// methods for https://github.com/jeetkanjani7/Parallel_NMS/blob/master/cpu/nms.cpp based nms


    struct boxes
    {
        float x,y,w,h,s;

    }typedef box;

    float IOUcalc(box b1, box b2);

    void nms_best(box *b, int count, bool *res);
{


  } /* namespace nms_utils */
} /* namespace open_ptrack */

#include <open_ptrack/nms_utils/nms.hpp>
#endif /* OPEN_PTRACK_NMS_UTILS_NMS_ */
