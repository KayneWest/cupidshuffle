#include <open_ptrack/nms_utils/nms.h>

#ifndef OPEN_PTRACK_NMS_UTILS_NMS_HPP_
#define OPEN_PTRACK_NMS_UTILS_NMS_HPP_

void open_ptrack::nms_utils::tvm_nms_cpu(std::vector<sortable_result>& boxes, MatF tvm_output, float cls_threshold, float nms_threshold, std::vector<sortable_result>& filterOutBoxes) {

    //ulsMatF(int cols, int rows, int channels)
    //at(int channel, int row, int col)
    //6, 322560, 1
    int valid_count = 0;
    for(int i = 0; i < tvm_output.getRows(); ++i){
      // we ONLY get people detections.
      if (*tvm_output.at(0, i, 1) >= cls_threshold && *tvm_output.at(0, i, 0) == 0){
        sortable_result res;
        res.index = valid_count;
        res.cls = *tvm_output.at(0, i, 0);//tvm_output.at(0, i, 0);
        res.probs = *tvm_output.at(0, i, 1);//tvm_output.at(1, i, 0);
        res.xmin = *tvm_output.at(0, i, 2);//tvm_output.at(2, i, 0);
        res.ymin = *tvm_output.at(0, i, 3);//tvm_output.at(3, i, 0);
        res.xmax = *tvm_output.at(0, i, 4);//tvm_output.at(4, i, 0);
        res.ymax = *tvm_output.at(0, i, 5);//tvm_output.at(5, i, 0);
        boxes.push_back(res);
        valid_count+=1;
      }
    }
    std::cout << "valid count: " << valid_count << std::endl;
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<sortable_result>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx].xmin, boxes[tmp_i].xmin );
            float inter_y1 = std::max( boxes[good_idx].ymin, boxes[tmp_i].ymin );
            float inter_x2 = std::min( boxes[good_idx].ymin, boxes[tmp_i].ymin );
            float inter_y2 = std::min( boxes[good_idx].ymax, boxes[tmp_i].ymax );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx].xmax - boxes[good_idx].xmin + 1) * (boxes[good_idx].ymax - boxes[good_idx].ymin + 1);
            float area_2 = (boxes[tmp_i].xmax - boxes[tmp_i].xmin + 1) * (boxes[tmp_i].ymax - boxes[tmp_i].ymin + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);           
            if( o <= nms_threshold )
                idx.push_back(tmp_i);
        }
    }
}

template <typename T>
static inline bool open_ptrack::nms_utils::SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
inline void  open_ptrack::nms_utils::GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    assert(score_index_vec.empty());
    // Generate index score pairs.
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}


// Do non maximum suppression given bboxes and scores.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    top_k: if not > 0, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
template <typename BoxType>
inline void open_ptrack::nms_utils::NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    assert(bboxes.size() == scores.size());

    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}

//template<typename _Tp> static inline
//double open_ptrack::nms_utils::jaccardDistance(const cv::Rect_<_Tp>& a, const cv::Rect_<_Tp>& b) {
//    _Tp Aa = a.area();
//    _Tp Ab = b.area();
//
//    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
//        // jaccard_index = 1 -> distance = 0
//        return 0.0;
//    }
//
//    double Aab = (a & b).area();
//    // distance = 1 - jaccard_index
//    return 1.0 - Aab / (Aa + Ab - Aab);
//}


template <typename T>
static inline float open_ptrack::nms_utils::rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(cv::jaccardDistance(a, b));
}

void open_ptrack::nms_utils::NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices)
{
    const float eta = 1.0f;
    const int top_k = 0;
    assert(bboxes.size() == scores.size());
    assert(score_threshold >= 0);
    assert(nms_threshold >= 0);
    assert(eta > 0);
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

void open_ptrack::nms_utils::opencv_nms( MatF tvm_output, float cls_threshold, float nms_threshold, std::vector<sortable_result>& boxes) {

    //ulsMatF(int cols, int rows, int channels)
    //at(int channel, int row, int col)
    //6, 322560, 1


    std::vector<cv::Rect> localBoxes;
    std::vector<float> localConfidences;
    std::vector<size_t> classIndices;
    std::vector<int> nmsIndices;

    int valid_count = 0;
    for(int i = 0; i < tvm_output.getRows(); ++i){
      if (*tvm_output.at(0, i, 1) >= cls_threshold && *tvm_output.at(0, i, 0) == 0){
        //sortable_result res;
        //res.index = valid_count;
        //res.cls = *tvm_output.at(0, i, 0);//tvm_output.at(0, i, 0);
        //res.probs = *tvm_output.at(0, i, 1);//tvm_output.at(1, i, 0);
        //res.xmin = *tvm_output.at(0, i, 2);//tvm_output.at(2, i, 0);
        //res.ymin = *tvm_output.at(0, i, 3);//tvm_output.at(3, i, 0);
        //res.xmax = *tvm_output.at(0, i, 4);//tvm_output.at(4, i, 0);
        //res.ymax = *tvm_output.at(0, i, 5);//tvm_output.at(5, i, 0);
        //boxes.push_back(res);
        cv::Rect rect;
        rect.x = *tvm_output.at(0, i, 2);//tvm_output.at(2, i, 0);
        rect.y = *tvm_output.at(0, i, 3);//tvm_output.at(3, i, 0);
        rect.width = *tvm_output.at(0, i, 4) - *tvm_output.at(0, i, 2);//tvm_output.at(4, i, 0);
        rect.height = *tvm_output.at(0, i, 5) - *tvm_output.at(0, i, 3);//tvm_output.at(5, i, 0);
        localBoxes.push_back(rect);
        localConfidences.push_back(*tvm_output.at(0, i, 1));
        classIndices.push_back(*tvm_output.at(0, i, 0));
        valid_count+=1;
      }
    }

    boxes.clear();
    if(localBoxes.size() == 0)
        return;

    NMSBoxes(localBoxes, localConfidences, cls_threshold, nms_threshold, nmsIndices);
    for (size_t i = 0; i < nmsIndices.size(); i++)
    {
        size_t idx = nmsIndices[i];
        sortable_result res;
        res.index = i;
        res.cls = 0; // automatically 0 for person
        res.probs = localConfidences[idx];
        res.xmin = localBoxes[idx].x;
        res.ymin = localBoxes[idx].y;
        res.xmax = localBoxes[idx].x + localBoxes[idx].width;
        res.ymax = localBoxes[idx].y + localBoxes[idx].height;
        boxes.push_back(res);
    }
}

\
        ///float *at(int channel, int row, int col){
         //   assert(m_data != NULL);
         //   assert(row < m_rows);
         //   assert(col < m_cols);
         //   assert(channel < m_channels);
         //   return m_data + (channel * m_rows * m_cols) + row * m_cols + col;


void open_ptrack::nms_utils::opencv_nms_mask( MatF conf_preds, MatF conf_indicies, MatF decoded_boxes, float cls_threshold, float nms_threshold, std::vector<sortable_result>& boxes) {

    //ulsMatF(int cols, int rows, int channels)
    //at(int channel, int row, int col)
    //6, 322560, 1


    std::vector<cv::Rect> localBoxes;
    std::vector<float> localConfidences;
    std::vector<size_t> classIndices;
    std::vector<int> nmsIndices;

    int valid_count = 0;
    for(int i = 0; i < conf_preds.getRows(); ++i){
      int class_value = static_cast<int>(*conf_indicies.at(0, 0, i));
      // just do people
      if (*conf_preds.at(0, i, index) >= cls_threshold && class_value == 0){
        cv::Rect rect;
        rect.x = *decoded_boxes.at(0, i, 0);
        rect.y = *decoded_boxes.at(0, i, 1);
        rect.width = *decoded_boxes.at(0, i, 2);// - *decoded_boxes.at(0, i, 0);
        rect.height = *decoded_boxes.at(0, i, 3);// - *decoded_boxes.at(0, i, 1);
        localBoxes.push_back(rect);
        localConfidences.push_back(*conf_preds.at(0, i, index));
        classIndices.push_back(*conf_indicies.at(0, 0, i)); // class_value
        valid_count+=1;
      }
    }

    boxes.clear();
    if(localBoxes.size() == 0)
        return;

    NMSBoxes(localBoxes, localConfidences, cls_threshold, nms_threshold, nmsIndices);
    for (size_t i = 0; i < nmsIndices.size(); i++)
    {
        size_t idx = nmsIndices[i];
        sortable_mask_result res;
        res.index = i;
        res.global_idx = idx;
        res.cls = 0; // automatically 0 for person
        res.probs = localConfidences[idx];
        res.xmin = localBoxes[idx].x;
        res.ymin = localBoxes[idx].y;
        //TODO nto sure if this is width or xmax
        res.xmax = localBoxes[idx].x + localBoxes[idx].width;
        res.ymax = localBoxes[idx].y + localBoxes[idx].height;
        // res.masks = conf_preds.at(0, i, 0)
        boxes.push_back(res);
    }
}

//https://docs.opencv.org/master/d4/db9/samples_2dnn_2object_detection_8cpp-example.html
#endif /* OPEN_PTRACK_NMS_UTILS_NMS_HPP_ */
