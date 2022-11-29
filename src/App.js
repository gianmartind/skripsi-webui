import './App.css'
import { useRef, useContext, useState } from 'react'

import ModelPicker from './components/model_picker'
import ImagePicker from './components/image_picker'
import ImageDisplay from './components/image_display'

import url from './config/urls'

import axios from 'axios'

function App() {
  const uniqueness = useRef(0)
  const consistency = useRef(0)
  const model = useRef(null)
  const image_upload = useRef(null)

  const [imageSrc, setImageSrc] = useState('./img_placeholder.png')
  const [imageName, setImageName] = useState('img_placeholder.png')
  const [totalWeight, setTotalWeight] = useState(0.0)

  const changeUniqueness = (x) => {
    uniqueness.current = x
  }

  const changeConsistency = (x) => {
    consistency.current = x
  }

  const changeModel = (x) => {
    model.current = x
  }

  const changeImage = (x) => {
    image_upload.current = x
  }

  const _uploadImage = () => {
    let formData = new FormData()
    formData.append('file', image_upload.current)
    axios.post(url.app.upload, formData)
      .then((res) => {
        console.log(res)
      })
  }

  const _detect = () => {
    if (model.current != null & image_upload.current != null) {
      let detect_param = {
        uniqueness: uniqueness.current,
        consistency: consistency.current,
        model: model.current
      }
      let formData = new FormData()
      formData.append('param', JSON.stringify(detect_param))
      formData.append('file', image_upload.current)
      axios.post(url.app.detect, formData)
        .then((res) => {
          console.log(res.data)
          let result = res.data[0]
          setImageSrc(`${url.BASE_URL}/static/detection/${result[2]}`)
          setImageName(result[0])
          setTotalWeight(result[1])
        })
    } else {
      alert('Image not selected!')
    }
  }

  return (
    <div className="row p-3">
      <div className="col-3">
        <div className="row container">
          <ImagePicker changeImage={changeImage} />
        </div>
        <div className="row container mt-3">
          <ModelPicker changeConsistency={changeConsistency} changeUniqueness={changeUniqueness} changeModel={changeModel} />
        </div>
        <div className="row container mt-3">
          <button className="btn btn-primary" onClick={_detect}>Detect</button>
        </div>
      </div>
      <div className="col-8">
        <div className="row">
          <ImageDisplay image_name={imageName} image_src={imageSrc} total_weight={totalWeight} />
        </div>
      </div>
    </div>
  )
}

export default App
