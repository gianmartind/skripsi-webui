import React from "react"

const ImageDisplay = ({ image_src, image_name, total_weight }) => {
    return (
        <div className="container row">
            <div className="container row border rounded">
                <img src={image_src} className="img-fluid" alt="detection result" />
            </div>
            <div className="container">
                <div className="float-start pe-4">
                    <div className="h3 fw-light">{image_name}</div>
                </div>
                <div className="float-start">
                    <div className="h3 fw-light">{total_weight}</div>
                </div>
            </div>
        </div>
    )
}

export default ImageDisplay