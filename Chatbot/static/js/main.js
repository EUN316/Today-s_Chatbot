// global variables
let product_id = null;
let question = null;
let state = 'SUCCESS';

// document.addEventListener('keydown', onClickAsEnter);


// functions
function Message(arg) {
  this.text = arg.text;
  this.message_side = arg.message_side;

  this.draw = function(_this) {
    return function() {
      let $message;
      $message = $($('.message_template').clone().html());
      $message.addClass(_this.message_side).find('.text').html(_this.text);
      $('.messages').append($message);

      return setTimeout(function() {
        return $message.addClass('appeared');
      }, 0);
    };
  }(this);
  return this;
}

function getMessageText() {
  let $message_input;
  $message_input = $('.message_input');
  return $message_input.val();
}

function sendMessage(text, message_side) {
  let $messages, message;
  $('.message_input').val('');
  $messages = $('.messages');
  message = new Message({
    text: text,
    message_side: message_side
  });
  message.draw();
  $messages.animate({
    scrollTop: $messages.prop('scrollHeight')
  }, 300);
}

// 시작 인사
function greet() {
  setTimeout(function() {
    return sendMessage("안녕하세요! 오늘의 집 리뷰 기반 Q&A 챗봇입니다 :)", 'left');
  }, 1000);

  setTimeout(function() {
    return sendMessage("문의를 남기고 싶은 상품ID를 입력해주세요.", 'left');
  }, 2000);
}

function onClickAsEnter(e) {
  if (e.keyCode === 13) {
    onSendButtonClicked()
  }
}

// 필터링
function setProductID(id) {
  if ((id !== null) && (id.replace(" ", "") != "")) {
    setTimeout(function() {
      return sendMessage(id + " 상품에 대한 문의를 남겨주세요.", 'left');
    }, 1000);

    return id;

  } else {
    setTimeout(function() {
      return sendMessage("해당 상품ID가 존재하지 않습니다. 다시 입력해주세요.", 'left');
    }, 1000);

    return null;
  }
}

function setQuestion(que) {
  if ((que !== null) && (que.replace(" ", "") != "")) {
    setTimeout(function() {
      return sendMessage("입력해주신 문의는 <" + que + "> 입니다.", 'left');
    }, 1000);

    return que;

  } else {
    setTimeout(function() {
      return sendMessage("문의를 다시 입력해주세요.", 'left');
    }, 1000);

    return null;
  }
}

function requestChat() {
  sendMessage('적절한 답변 및 리뷰를 분석 중입니다. 잠시만 기다려주세요!', 'left');
  $.ajax({
    type: "POST",
    contentType: "application/x-www-form-urlencoded; charset=UTF-8",
    url: "http://0.0.0.0:8080/process",
    data: {
      input_id: product_id,
      input_que: question
    },
    dataType: "json",
    success: function(data) {
      setTimeout(function() {
        return sendMessage('상품ID: ' + data['review_']['prod_id'][0], 'left');
      }, 500);
      setTimeout(function() {
        return sendMessage('상품명: ' + data['review_']['prod_name'][0], 'left');
      }, 1500);
      // 유사한 질문 1개
      setTimeout(function() {
        return sendMessage('유사 질문: ' + data['qna_']['question'][0], 'left');
      }, 2500);
      setTimeout(function() {
        return sendMessage('답변: ' + data['qna_']['answer'][0], 'left');
      }, 4000);

      for (i in data['review_']['comment']) {
        (function(x) {
          setTimeout(function() {
            var xt = parseInt(x)+1;
            sendMessage('리뷰 ' + xt+ ': ' + data['review_']['comment'][x], 'left');
            sendMessage('이미지 url' + xt + ': ' + data['review_']['image_url'][x], 'left');
          }, 5500 + 1000 * x);
        })(i);
      };

      setTimeout(function() {
        return sendMessage('이용해주셔서 감사합니다.', 'left');
      }, 13000);
      // 리셋
      product_id = null;
      question = null;
      setTimeout(function() {
        return greet();
      }, 15000);
    },

    error: function(xtr, status, error) {
      sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'left');
      // 리셋
      product_id = null;
      question = null;
      setTimeout(function() {
        return greet();
      }, 1000);
    }
  });
}


function onSendButtonClicked() {
  let messageText = getMessageText();
  sendMessage(messageText, 'right');

  if (product_id === null) {
    product_id = setProductID(messageText);
  } else if (question === null) {
    question = setQuestion(messageText);
  }

  if ((product_id !== null) && (question !== null)) {
    // setTimeout(function() {
    //   return sendMessage('처리 중입니다. 잠시만 기다려주세요.', 'left');
    // }, 1000);
    setTimeout(function() {
      return requestChat();
    }, 2000);
  }
}
