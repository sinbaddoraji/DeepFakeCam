#pragma once
#include <QDialog>
#include <QImage>
#include <QRect>

class FacePreviewDialog : public QDialog {
    Q_OBJECT
public:
    FacePreviewDialog(const QImage& image, const QRect& faceRect, QWidget* parent = nullptr);
    QRect getCorrectedRect() const;

private:
    class FacePreviewWidget* previewWidget;
    QRect correctedRect;
}; 