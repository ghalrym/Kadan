import { useState } from "react";
import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";
import { assetUrl } from "./api";
import { Markdown } from "./components";
import type { Asset, Deck, Slide } from "./types";

export function SlidePreview({
  slide,
  agentKey,
}: {
  slide: Slide;
  agentKey: string;
}) {
  return (
    <div className={`slide-preview layout-${slide.layout}`}>
      <div className="slide-copy">
        <h3>{slide.title}</h3>
        <Markdown text={slide.body} />
      </div>
      {slide.image_asset_id && (
        <img
          src={assetUrl(agentKey, slide.image_asset_id)}
          alt={slide.title || "Slide image"}
        />
      )}
    </div>
  );
}
export function DeckEditor({
  deck,
  onChange,
  images,
  agentKey,
}: {
  deck: Deck;
  onChange: (deck: Deck) => void;
  images: Asset[];
  agentKey: string;
}) {
  const [index, setIndex] = useState(0);
  const slide = deck.slides[index];
  const update = (patch: Partial<Slide>) =>
    onChange({
      ...deck,
      slides: deck.slides.map((item, i) =>
        i === index ? { ...item, ...patch } : item,
      ),
    });
  const add = () => {
    onChange({
      ...deck,
      slides: [
        ...deck.slides,
        {
          id: crypto.randomUUID(),
          layout: "title",
          title: "",
          body: "",
          notes: "",
          image_asset_id: null,
        },
      ],
    });
    setIndex(deck.slides.length);
  };
  const move = (direction: number) => {
    const slides = [...deck.slides];
    [slides[index], slides[index + direction]] = [
      slides[index + direction],
      slides[index],
    ];
    onChange({ ...deck, slides });
    setIndex(index + direction);
  };
  return (
    <div className="deck-editor">
      <div className="slide-strip">
        {deck.slides.map((item, i) => (
          <button
            type="button"
            key={item.id}
            className={`slide-tab ${i === index ? "selected" : ""}`}
            onClick={() => setIndex(i)}
          >
            <span>{i + 1}</span>
            {item.title || "Untitled slide"}
          </button>
        ))}
        <button
          type="button"
          className="small-button"
          onClick={add}
          disabled={deck.slides.length >= 100}
        >
          <Plus size={16} />
          Add slide
        </button>
      </div>
      {slide ? (
        <div className="slide-workspace">
          <SlidePreview slide={slide} agentKey={agentKey} />
          <div className="slide-fields">
            <div className="field-pair">
              <label>
                Layout
                <select
                  value={slide.layout}
                  onChange={(event) =>
                    update({ layout: event.target.value as Slide["layout"] })
                  }
                >
                  <option value="title">Title</option>
                  <option value="text">Text</option>
                  <option value="split">Split</option>
                  <option value="image">Image</option>
                </select>
              </label>
              <label>
                Image
                <select
                  value={slide.image_asset_id || ""}
                  onChange={(event) =>
                    update({ image_asset_id: event.target.value || null })
                  }
                >
                  <option value="">No image</option>
                  {images.map((image) => (
                    <option key={image.id} value={image.id}>
                      {image.title}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label>
              Slide title
              <input
                value={slide.title}
                onChange={(event) => update({ title: event.target.value })}
              />
            </label>
            <label>
              Body
              <textarea
                rows={5}
                value={slide.body}
                onChange={(event) => update({ body: event.target.value })}
              />
            </label>
            <label>
              Speaker notes
              <textarea
                rows={3}
                value={slide.notes}
                onChange={(event) => update({ notes: event.target.value })}
              />
            </label>
            <div className="button-row">
              <button
                type="button"
                className="small-button"
                disabled={index === 0}
                onClick={() => move(-1)}
              >
                <ArrowUp size={16} />
                Move up
              </button>
              <button
                type="button"
                className="small-button"
                disabled={index === deck.slides.length - 1}
                onClick={() => move(1)}
              >
                <ArrowDown size={16} />
                Move down
              </button>
              <button
                type="button"
                className="small-button danger"
                onClick={() => {
                  onChange({
                    ...deck,
                    slides: deck.slides.filter((_, i) => i !== index),
                  });
                  setIndex(Math.max(0, index - 1));
                }}
              >
                <Trash2 size={16} />
                Delete slide
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="empty">
          <p>No slides yet.</p>
          <button type="button" onClick={add}>
            <Plus size={16} />
            Add slide
          </button>
        </div>
      )}
    </div>
  );
}
