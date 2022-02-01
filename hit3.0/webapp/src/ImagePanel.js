import { Typography } from "@material-ui/core";
import React from "react";
import "./ImagePanel.css"


class ImagePanel extends React.Component {
    constructor(props) {
      super(props);
      console.log(props);
      this.state = {
        imgUrl: props.imgUrl,
        questionStr: props.questionStr,
      };
    }
    render() {
        console.log(this.state.imgUrl);
        return (
            <div className="outer">
                <div className="text">
                    <Typography variant="h5" align="center">
                       <b> Question: </b> {this.state.questionStr}
                    </Typography>
                </div>
                <div className="inner">
                    <img src={this.state.imgUrl} alt="blah"/>
                </div>
            </div>
        );
    }
}
export default ImagePanel;