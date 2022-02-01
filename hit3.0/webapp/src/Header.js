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
            TODO: Elias: add instructions here 
          </Typography>
        </Dialog>
      </div>
    )
  }
}
export default Header
