import './App.css'
import { useRef, useState } from 'react'

import ModelPicker from './components/model_picker'
import ImagePicker from './components/image_picker'
import ImageDisplay from './components/image_display'
import ResultList from './components/result_list'

import url from './config/urls'

import { Blocks } from  'react-loader-spinner'
import axios from 'axios'

function App() {
  const uniqueness = useRef(0)
  const consistency = useRef(0)
  const model = useRef(null)
  const image_upload = useRef(null)

  const [imageSrc, setImageSrc] = useState('./img_placeholder.png')
  const [imageName, setImageName] = useState('img_placeholder.png')
  const [totalWeight, setTotalWeight] = useState(0.0)

  const [loading, setLoading] = useState(false)
  const data = ['', '', 'img_placeholder.png']
  const [resultList, setResultList] = useState([data])
  const [totalTime, setTotalTime] = useState(null)

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

  const _identify = () => {
    setLoading(true)
    if (model.current != null & image_upload.current != null) {
      let detect_param = {
        uniqueness: uniqueness.current,
        consistency: consistency.current,
        model: model.current
      }
      let formData = new FormData()
      formData.append('param', JSON.stringify(detect_param))
      formData.append('file', image_upload.current)
      axios.post(url.app.identify, formData)
        .then((res) => {
          setResultList(res.data.list)
          setLoading(false)
          setTotalTime(`${res.data.total_time}s`)
        })
    } else {
      alert('Image not selected!')
    }
  }

  const changeImageDisplay = (data) => {
    setImageName(data[0])
    setTotalWeight(data[1])
    setImageSrc(`${url.BASE_URL}/static/detection/${data[2]}`)
  }
  
  return (
    <div className="container-fluid p-3">
      <div className='row'>
        <div className="col-3">
          <div className="container-fluid">
            <ImagePicker changeImage={changeImage} />
          </div>
          <div className="container-fluid mt-3">
            <ModelPicker changeConsistency={changeConsistency} changeUniqueness={changeUniqueness} changeModel={changeModel} />
          </div>
          <div className="container-fluid mt-3">
            <button className="btn btn-block btn-primary" onClick={_identify}>Identify</button>
            {loading ? (
              <Blocks
                visible={true}
                width={38}
                height={38}
                ariaLabel="blocks-loading"
                wrapperStyle={{}}
              />
            ) : (
              <span className="fw-lighter container">{totalTime}</span>
            )}
          </div>
        </div>
        <div className="col-9">
          <div className="container-fluid">
            <ImageDisplay image_name={imageName} image_src={imageSrc} total_weight={totalWeight} />
          </div>
          <ResultList result_list={resultList} changeImageDisplay={changeImageDisplay}/>
        </div>
      </div>
    </div>
  )
}

export default App
