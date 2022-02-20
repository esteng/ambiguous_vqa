import React, {PureComponent} from "react";

import AppBar from '@material-ui/core/AppBar';
import Button from '@material-ui/core/Button';
import DialogTitle from '@material-ui/core/DialogTitle';
import Dialog from '@material-ui/core/Dialog';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';

import "./Header.css"

class Header extends PureComponent {
  constructor(props){
    super(props);

    this.state = {
      dialogOpen: false,
    };
  }


  handleDialogOpen = () => {
    this.setState({dialogOpen: true});
  }

  handleDialogClose = () => {
    this.setState({dialogOpen: false});
  }

  render() {
    return (
      <div>
        <AppBar position="static">
          <Toolbar>
            <Typography className='appbarSpace' variant="h6" color="inherit">
              Annotator Disagreement 
            </Typography>


            <Button color="inherit" onClick={this.handleDialogOpen}>
              Show Instructions
            </Button>
          </Toolbar>
        </AppBar>

        <Dialog open={this.state.dialogOpen} onClose={this.handleDialogClose} aria-labelledby="simple-dialog-title">
          <DialogTitle>Instructions</DialogTitle>
          <Typography className='dialogText'>
            These instructions are explained with examples <a href="https://youtu.be/qvmrL7JAlgk" target="_blank">in this video</a>.
            In each example, you'll see an image, a question about the image, and a set of answers. 
            We are interested in figuring out what question the answers are actually answering.<br/> 
            <br/>
            Often, all the answers are answers to the same question; these cases aren't interesting to us, and you can skip them. 
            But other times, the meaning of the question isn't totally clear, and so there might be two or more distinct groups of answers. 
            <br/>
            <br/>
            There are two tasks involved in performing this HIT. 
            <ol>
              <li> you need to determine whether there are any groups of answers, and if there are, use the drag-and-drop interface to group them.</li>
              <li>Then you will be asked to rewrite the question so that it more clearly matches each answer group.  </li>
            </ol>
            This rewrite should be minimal -- try to make the smallest change that you can to the original question.
            Your goal should be to rewrite the question so that if someone else answered the question, they would only answer with one of the answers in the group, and not with an answer from any other group.
            <br/>
            <br/>
            Something to note: what is important here is <b>not</b> that all of the answers in the group be the same. 
            For example, you can have opposite answers in the same group, e.g. for the question, "Did you read the instructions?", "yes" and "no" answer the same question, and would be in the same group. 
            <br/>
            <br/>

            Interface Notes: 
            <ul>
              <li> If you skip an example, you can provide a reason for skipping in the text box. </li>
              <li> If there are more answers than the width of your screen (usually, 3 or 4 examples) then they will overflow to the next line, and you may have to scroll horizontally to see all  the answers. </li>
              <li>Spam answers can be removed using the "Delete" button. Deleting all the answers in a group or moving all the answers to another group will delete that column. You can add additional groups using the bottom at the top of the interface.</li>
              <li>Any changes can be erased by hitting the "reset" button.</li>
            </ul>
          </Typography>
        </Dialog>
      </div>
    )
  }
}
export default Header
