/**
 * Added by Chaobin Tang <cbtchn@gmail.com>
 * Added on 2013-05-30
 *
**/

(function () {

  "use strict";

  /**
   * External name
  **/
  var snail = window.snail || {};
  window.snail = snail;

  snail.ui = {};


  snail.ui.print = function (chars, width) {
    var chars = chars.toString();
    document.write(chars);
  }

  snail.ui.println = function (chars) {
    document.writeln(chars + '</br>');
  }

  snail.ui.draw = function (matrix) {
    /**
     * Print the pattern into ducument.
    **/
    var size = matrix.length;

    for (var x = 0; x < size; x++) {

      for (var y = 0; y < size; y++) {
        var n = matrix[x][y];
        if (n < 0) {
          continue;
        }
        snail.ui.pointToDiv(n, x, y);
      }
      snail.ui.println('');
    }
  }

  snail.ui.makeIdFromXY = function (x, y) {
    return 'p' + x + y;
  }

  snail.ui.pointToDiv = function (n, x, y) {
    /**
     * IMPORTANT! This function isn't something permenant.
     *
     */
    var _id = snail.ui.makeIdFromXY(x, y);

    var div = '<div class="point" id=' + _id + '>' + n + '</div>';
    
    snail.ui.print(div);
  }

  snail.ui.breathe = function (divPoint) {
    /**
     * Replace something inside a located
     * point div to make it look funny.
     * Since being too fast makes it impossible
     * for human-eyes to see the transition,
     * this function should be instead called
     * using setInterval() with a reasonable
     * amount of pause.
     *
    **/
    var oldCtt = divPoint.innerHTML;
    var newCtt = '*';
    divPoint.innerHTML = newCtt;
  }

  snail.ui.animate = function (pattern) {
    /**
     * Traverse the pattern, using the coordinates
     * to animate the pattern in one of the many funny
     * ways.
     *
    **/

    var i = 1;
    setInterval(function () {
      if (i > pattern.pow) {
        clearInterval(window.animator);
        return;
      }
      var point = pattern._map[i];
      var _id = snail.ui.makeIdFromXY(point.x, point.y);
      var divPoint = document.getElementById(_id);
      snail.ui.breathe(divPoint);
      i++;
    }, 1000);
  }

  /**
   * Class used to calculate the snail-like spiral pattern.
  **/
  snail.Snail = function (size) {

    var that = this;

    that.size = size;

    that.center = (function () {
      /**
       * The index of the center value
       * in a 0-indexed array.
      **/
      return Math.ceil( (that.size - 1) / 2 );
    })();

    that.isOdd = function (n) {
      return ( (n & 1) == 1 );
    }

    that.corrs = function () {
      /**
       * Return array of odd numbers.
       * e.g. 
       *-
       * 1 -> [1]
       * 3 -> [1, 3]
       * 5 -> [1, 3, 5]
       * 7 -> [1, 3, 5, 7]
      **/

      var corrs = [];
      
      var i = 0;
      while (i <= that.size) {
        if ( that.isOdd(i) ) {
          corrs.push(i);
        }
        i += 1;
      }

      return corrs;
    }

    that.zeros = function () {
      /**
       * Calculate a 2-D array pre-filled
       * with 0.
      **/
      var Ys = [];
      for (var i = 0; i < that.size; i++) {
        var Xs = [];
        for (var j = 0; j < that.size; j++) {
          Xs.push(0);
        }
        Ys.push(Xs);
      }
      return Ys;

    }

    that.distance = function (axis) {
      /**
       * Return the distance to the center
       * along one certain axis, be it
       * either x or y.
      **/
      return Math.abs(axis - that.center) - 1;
    }

    that.corrByPoint = function (x, y) {
      /**
       * Return the corresponding correlation
       * given the coordinates in the matrix.
      **/
      
      var distance_x = that.distance(x);
      var distance_y = that.distance(y);

      var corrIndex = distance_x;
      if ( distance_x < distance_y ) {
        corrIndex = distance_x + (distance_y - distance_x);
      }

      return that._corrs[corrIndex + 1];
    }

    that.draw = function () {
      /**
       *
       * IMPORTANT! This is the formula that
       * calculates the number based on an observation
       * that number value decreases along the direction
       * spiral to the center of the pattern.
       *
      **/
      if (that._map.length > 0) { // Means already calculated.
        return;
      }

      var n = 0, pow, topLeft, bottomLeft, bottomRight;

      for (var x = 0; x < that.size; x++) {
        for (var y = 0, last = that.tail; y < that.size; y++) {

          var corr = that.corrByPoint(x, y);
          /*
          if (y != last) {
            snail.ui.print(corr + ' - ');
          } else {
            snail.ui.print(corr);
          }*/
          pow = Math.pow(corr, 2);
          topLeft = pow - (corr - 1);
          bottomLeft = topLeft - (corr - 1);
          bottomRight = bottomLeft - (corr - 1);

          n = ( topLeft - x + y);
          if ( x > that.center ) {
            if ( y > (that.tail - x) && y < ( x + 1)) {
              n = bottomLeft - y + (that.tail - x);
            }
          }

          if ( y > that.center ) {
            
            if ( x > (that.tail - y) && x < y ) { // cause it is one-element-less than bottom-x-axis.
              n = bottomRight - (that.tail - x) + (that.tail - y);
            }
          }

          if (n > that.pow) {
            n = -1;
          }
          that._zeros[x][y] = n;
          // Pinpoint this number with its coordinates
          that._map[n] = {x:x, y:y};
        }
      }

    }

    that.prepare = function () {
      /**
       *
       * Called before draw().
       * This is made to be lazy.
       * 
       * IMPORTANT! When the size is an even number,
       * the size is incremented by 1 to make it odd.
       * This is because the pattern of the size of an
       * odd number has the pattern of that of (size-1)
       * contained, only it has an extra top and right
       * 'border', and there are instead filled with
       * '-1' to indicate their uselessness when feeded
       * to the output.
       *
      **/
      if ( that.pow == undefined ) {

        that.pow = Math.pow(that.size, 2);

        if (! that.isOdd(that.size) ) {
          that.pow = Math.pow(that.size, 2);
          that.size += 1;
        }

        that.tail = that.size - 1;
      }

      if ( that._corrs == undefined ) {
        that._corrs = that.corrs();
        that._zeros = that.zeros(); // Computed results;
        that._map = {};
      }
    }
  }

})();