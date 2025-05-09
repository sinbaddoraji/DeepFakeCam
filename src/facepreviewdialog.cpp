#include "facepreviewdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPainter>
#include <QMouseEvent>

class FacePreviewWidget : public QWidget {
public:
    FacePreviewWidget(const QImage& img, const QRect& rect, QWidget* parent = nullptr)
        : QWidget(parent), image(img), faceRect(rect), dragging(false), resizing(false) {
        setMinimumSize(300, 300);
    }
    QRect getRect() const { return faceRect; }
protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        QImage scaled = image.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        p.drawImage(rectForImage(scaled), scaled);
        QRect r = mapRectToWidget(faceRect, scaled);
        p.setPen(QPen(Qt::red, 2));
        p.drawRect(r);
        p.setBrush(QColor(255,0,0,80));
        p.drawRect(r);
    }
    void mousePressEvent(QMouseEvent* e) override {
        QRect r = mapRectToWidget(faceRect, image.scaled(size(), Qt::KeepAspectRatio));
        if (r.contains(e->pos())) {
            dragging = true;
            dragOffset = e->pos() - r.topLeft();
        } else if (QRect(r.bottomRight()-QPoint(10,10), QSize(20,20)).contains(e->pos())) {
            resizing = true;
            resizeStart = e->pos();
            origRect = r;
        }
    }
    void mouseMoveEvent(QMouseEvent* e) override {
        if (dragging) {
            QPoint newTopLeft = e->pos() - dragOffset;
            QRect r = mapRectToWidget(faceRect, image.scaled(size(), Qt::KeepAspectRatio));
            QRect newRect(newTopLeft, r.size());
            faceRect = mapRectFromWidget(newRect, image.scaled(size(), Qt::KeepAspectRatio));
            update();
        } else if (resizing) {
            int dx = e->pos().x() - resizeStart.x();
            int dy = e->pos().y() - resizeStart.y();
            QRect r = origRect;
            r.setWidth(std::max(10, r.width() + dx));
            r.setHeight(std::max(10, r.height() + dy));
            faceRect = mapRectFromWidget(r, image.scaled(size(), Qt::KeepAspectRatio));
            update();
        }
    }
    void mouseReleaseEvent(QMouseEvent*) override {
        dragging = false;
        resizing = false;
    }
private:
    QImage image;
    QRect faceRect;
    bool dragging, resizing;
    QPoint dragOffset, resizeStart;
    QRect origRect;
    QRect rectForImage(const QImage& img) const {
        QSize s = img.size();
        QSize ws = size();
        int x = (ws.width() - s.width())/2;
        int y = (ws.height() - s.height())/2;
        return QRect(x, y, s.width(), s.height());
    }
    QRect mapRectToWidget(const QRect& r, const QImage& img) const {
        QRect imgRect = rectForImage(img);
        double sx = double(imgRect.width()) / image.width();
        double sy = double(imgRect.height()) / image.height();
        return QRect(
            imgRect.left() + int(r.left()*sx),
            imgRect.top() + int(r.top()*sy),
            int(r.width()*sx), int(r.height()*sy)
        );
    }
    QRect mapRectFromWidget(const QRect& r, const QImage& img) const {
        QRect imgRect = rectForImage(img);
        double sx = double(image.width()) / imgRect.width();
        double sy = double(image.height()) / imgRect.height();
        return QRect(
            int((r.left()-imgRect.left())*sx),
            int((r.top()-imgRect.top())*sy),
            int(r.width()*sx), int(r.height()*sy)
        );
    }
};

FacePreviewDialog::FacePreviewDialog(const QImage& image, const QRect& faceRect, QWidget* parent)
    : QDialog(parent), correctedRect(faceRect) {
    setWindowTitle("Preview and Correct Face");
    QVBoxLayout* layout = new QVBoxLayout(this);
    previewWidget = new FacePreviewWidget(image, faceRect, this);
    layout->addWidget(previewWidget);
    QHBoxLayout* btnLayout = new QHBoxLayout();
    QPushButton* okBtn = new QPushButton("OK");
    QPushButton* cancelBtn = new QPushButton("Cancel");
    btnLayout->addStretch();
    btnLayout->addWidget(okBtn);
    btnLayout->addWidget(cancelBtn);
    layout->addLayout(btnLayout);
    connect(okBtn, &QPushButton::clicked, this, [this]() {
        correctedRect = previewWidget->getRect();
        accept();
    });
    connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
    setMinimumSize(400, 400);
}
QRect FacePreviewDialog::getCorrectedRect() const {
    return correctedRect;
} 