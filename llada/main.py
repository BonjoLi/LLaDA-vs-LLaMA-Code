import os
import gc
import argparse
import time

# Avoid loading all models when importing helpers from app.py.
os.environ.setdefault("LLADA_SKIP_PRELOAD", "1")


def parse_args():
    """Parse CLI args to choose which model and GPU to use."""
    parser = argparse.ArgumentParser(
        description="Run single-model LLaDA demo (0: diffusion, 1: autoregressive)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--0",
        dest="model_choice",
        action="store_const",
        const="diffusion",
        help="Use diffusion model (LLaDA 8B).",
    )
    group.add_argument(
        "--1",
        dest="model_choice",
        action="store_const",
        const="autoregressive",
        help="Use autoregressive model (Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Set CUDA device before importing app (which detects device at import time).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import gradio as gr
import app  # noqa: E402


def create_chatbot_demo_single_model(selected_model):
    """Create a Gradio demo that runs with a single, pre-selected model."""
    responsive_css = app.css + """
    /* Make the chat area fill the viewport so there is no empty space below. */
    :root { --chat-vertical-padding: 240px; }
    .gradio-container { min-height: 100vh; }
    #chatbot-ui {
        height: calc(100vh - var(--chat-vertical-padding)) !important;
        max-height: calc(100vh - var(--chat-vertical-padding));
        min-height: 480px;
    }
    #chatbot-ui .wrap { height: 100%; }
    """
    with gr.Blocks() as demo:
        gr.HTML(f"<style>{responsive_css}</style>")
        chat_history = gr.State([])

        chatbot_ui = gr.Chatbot(
            height=None,
            show_label=False,
            render_markdown=True,
            elem_id="chatbot-ui",
            type="messages",
        )

        with gr.Row():
            user_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                show_label=False,
            )
            send_btn = gr.Button("Send")

        with gr.Row():
            clear_btn = gr.Button("Clear Conversation")

        # HELPER FUNCTIONS
        def add_message(history, message, response):
            history = history.copy()
            history.append([message, response])
            return history

        def user_message_submitted(message, history):
            if not message.strip():
                history_for_display = app.format_chatbot_display(history)
                return history, history_for_display, ""

            history = add_message(history, message, None)
            history_for_display = app.format_chatbot_display(history)
            message_out = ""
            return history, history_for_display, message_out

        def bot_response(history):
            if not history:
                return history, []

            # Clear any previous stop signal
            app.GENERATION_STOP_EVENT.clear()

            last_user_message = history[-1][0]

            try:
                components = app.load_model_components(selected_model)
                tokenizer_instance = components["tokenizer"]
                model_instance = components["model"]
                mode = components["mode"]
                model_device = components.get("device", app.device)

                messages = app.format_chat_history(history[:-1])
                messages.append({"role": "user", "content": last_user_message})

                if mode == "diffusion":
                    vis_stream = app.generate_response_with_visualization(
                        model_instance,
                        tokenizer_instance,
                        model_device,
                        messages,
                        gen_length=app.DEFAULT_GEN_LENGTH,
                        steps=app.DEFAULT_STEPS,
                        constraints=None,
                        temperature=app.DEFAULT_TEMPERATURE,
                        cfg_scale=app.DEFAULT_CFG_SCALE,
                        block_length=app.DEFAULT_BLOCK_LENGTH,
                        remasking=app.DEFAULT_REMASKING,
                    )

                    for state, partial_text, is_final in vis_stream:
                        if app.GENERATION_STOP_EVENT.is_set():
                            return
                            
                        display_text = partial_text if is_final else app.state_to_text(state)
                        history[-1][1] = display_text

                        history_for_display = app.format_chatbot_display(history)
                        yield history, history_for_display

                        if not is_final and app.DEFAULT_VISUALIZATION_DELAY > 0:
                            time.sleep(app.DEFAULT_VISUALIZATION_DELAY)
                else:
                    text_stream = app.autoregressive_response_stream(
                        model_instance,
                        tokenizer_instance,
                        messages,
                        temperature=app.DEFAULT_TEMPERATURE,
                        device_override=model_device,
                    )

                    for partial_text, is_final in text_stream:
                        if app.GENERATION_STOP_EVENT.is_set():
                            return
                            
                        history[-1][1] = partial_text
                        history_for_display = app.format_chatbot_display(history)
                        yield history, history_for_display

                        if not is_final and app.DEFAULT_VISUALIZATION_DELAY > 0:
                            time.sleep(app.DEFAULT_VISUALIZATION_DELAY)

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)

                history[-1][1] = error_msg
                history_for_display = app.format_chatbot_display(history)
                yield history, history_for_display

        def clear_conversation():
            app.GENERATION_STOP_EVENT.set()
            gc.collect()
            if app.torch.cuda.is_available():
                app.torch.cuda.empty_cache()
            return [], [], ""

        # Event handles
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input],
        )

        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input],
        )

        bot_res_submit = msg_submit.then(
            fn=bot_response,
            inputs=[chat_history],
            outputs=[chat_history, chatbot_ui],
        )

        bot_res_click = send_click.then(
            fn=bot_response,
            inputs=[chat_history],
            outputs=[chat_history, chatbot_ui],
        )

        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history, chatbot_ui, user_input],
            queue=False
        )

    return demo


if __name__ == "__main__":
    # args already parsed above (before app import) to set CUDA_VISIBLE_DEVICES.
    selected_model = (
        app.LLADA_MODEL_NAME if args.model_choice == "diffusion" else app.LLAMA3_MODEL_NAME
    )
    print(f"Using GPU: {args.gpu}", flush=True)
    print(f"Preloading model: {selected_model} ...", flush=True)
    app.load_model_components(selected_model)
    print("Model ready. Starting Gradio...", flush=True)

    demo = create_chatbot_demo_single_model(selected_model)
    demo.queue().launch(share=True)

