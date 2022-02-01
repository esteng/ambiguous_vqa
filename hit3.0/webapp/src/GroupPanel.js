import { TextField, Typography } from "@material-ui/core";
import React, { Component } from "react";
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

import "./GroupPanel.css"


// fake data generator
const getItems = (count, offset = 0) =>
  Array.from({ length: count }, (v, k) => k).map(k => ({
    id: `item-${k + offset}-${new Date().getTime()}`,
    content: `item ${k + offset}`
  }));

const reorder = (list, startIndex, endIndex) => {
  const result = Array.from(list);
  const [removed] = result.splice(startIndex, 1);
  result.splice(endIndex, 0, removed);

  return result;
};

/**
 * Moves an item from one list to another list.
 */
const move = (source, destination, droppableSource, droppableDestination) => {
  const sourceClone = Array.from(source);
  const destClone = Array.from(destination);
  const [removed] = sourceClone.splice(droppableSource.index, 1);

  destClone.splice(droppableDestination.index, 0, removed);

  const result = {};
  result[droppableSource.droppableId] = sourceClone;
  result[droppableDestination.droppableId] = destClone;

  return result;
};
const grid = 8;

const getItemStyle = (isDragging, draggableStyle) => ({
  // some basic styles to make the items look a bit nicer
  userSelect: "none",
  padding: grid * 2,
  margin: `0 0 ${grid}px 0`,

  // change background colour if dragging
  background: isDragging ? "lightgreen" : "grey",

  // styles we need to apply on draggables
  ...draggableStyle
});
const getListStyle = isDraggingOver => ({
  background: isDraggingOver ? "lightblue" : "lightgrey",
  padding: grid,
  width: 250
});

class GroupPanel extends Component {
    constructor(props) {
      super(props);
      console.log(props);
      this.state = {
        answerGroups: props.answerGroups,
        answerQuestions: props.answerQuestions,
        fieldIds: [...props.answerGroups.keys()],
        items: props.answerGroups,
      };
    }
    onDragEnd = result => {
        const { source, destination } = result;
        console.log("this in onDragEnd", this);
        // dropped outside the list
        if (!destination) {
          return;
        }
        const sInd = +source.droppableId;
        const dInd = +destination.droppableId;
    
        if (sInd === dInd) {
          const items = reorder(this.state.items[sInd], source.index, destination.index);
          const newStateItems = [...this.state.items];
          newStateItems[sInd] = items;
          this.setState({items: newStateItems});
        } else {
          const result = move(this.state.items[sInd], this.state.items[dInd], source, destination);
          const newStateItems = [...this.state.items];
          newStateItems[sInd] = result[sInd];
          newStateItems[dInd] = result[dInd];
    
          this.setState({items: newStateItems.filter(group => group.length)});
        }
    }

    render(){
        console.log("this in render", this)
        return (
            <div>
              <button
                type="button"
                onClick={() => {
                  this.setState({items: [...this.state.items, []]});
                }}
              >
                Add new group
              </button>
              <button
                type="button"
                onClick={() => {
                  this.setState({items: [...this.state.items, getItems(1)]});
                }}
              >
                Add new item
              </button>
              <div style={{ display: "flex" }}>
                <DragDropContext onDragEnd={this.onDragEnd}>
                  {this.state.items.map((el, ind) => (
                    <div>
                        {/* {console.log("this.state.fieldIds[ind]", this.state.fieldIds[ind])} */}
                        {/* {console.log("this.state.answerQuestions[ind]", this.state.answerQuestions[ind])} */}
                        <TextField id={this.state.fieldIds[ind]} label={this.state.answerQuestions[ind]} variant="outline" />
                        <Droppable key={ind} droppableId={`${ind}`}>
                        {(provided, snapshot) => (
                            <div
                            ref={provided.innerRef}
                            style={getListStyle(snapshot.isDraggingOver)}
                            {...provided.droppableProps}
                            >
                            {el.map((item, index) => (
                                <Draggable
                                key={item.id}
                                draggableId={item.id}
                                index={index}
                                >
                                {(provided, snapshot) => (
                                    <div
                                    ref={provided.innerRef}
                                    {...provided.draggableProps}
                                    {...provided.dragHandleProps}
                                    style={getItemStyle(
                                        snapshot.isDragging,
                                        provided.draggableProps.style
                                    )}
                                    >
                                    <div
                                        style={{
                                        display: "flex",
                                        justifyContent: "space-around"
                                        }}
                                    >
                                        {item.content}
                                        <button
                                        type="button"
                                        onClick={() => {
                                            const newStateItems = [...this.state.items];
                                            newStateItems[ind].splice(index, 1);
                                            this.setState(
                                            {items: newStateItems.filter(group => group.length)}
                                            );
                                        }}
                                        >
                                        delete
                                        </button>
                                    </div>
                                    </div>
                                )}
                                </Draggable>
                            ))}
                            {provided.placeholder}
                            </div>
                        )}
                        </Droppable>
                    </div>
                  ))}
                </DragDropContext>
              </div>
            </div>
          );
    }
    
}
export default GroupPanel;