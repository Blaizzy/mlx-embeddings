# typed: false
# frozen_string_literal: true

require "utils"

describe Utils do
  describe ".deconstantize" do
    it "removes the rightmost segment from the constant expression in the string" do
      expect(described_class.deconstantize("Net::HTTP")).to eq("Net")
      expect(described_class.deconstantize("::Net::HTTP")).to eq("::Net")
      expect(described_class.deconstantize("String")).to eq("")
      expect(described_class.deconstantize("::String")).to eq("")
    end

    it "returns an empty string if the namespace is empty" do
      expect(described_class.deconstantize("")).to eq("")
      expect(described_class.deconstantize("::")).to eq("")
    end
  end

  describe ".demodulize" do
    it "removes the module part from the expression in the string" do
      expect(described_class.demodulize("Foo::Bar")).to eq("Bar")
    end

    it "returns the string if it does not contain a module expression" do
      expect(described_class.demodulize("FooBar")).to eq("FooBar")
    end

    it "returns an empty string if the namespace is empty" do
      expect(described_class.demodulize("")).to eq("")
      expect(described_class.demodulize("::")).to eq("")
    end
  end

  describe ".pluralize" do
    it "combines the stem with the default suffix based on the count" do
      expect(described_class.pluralize("foo", 0)).to eq("foos")
      expect(described_class.pluralize("foo", 1)).to eq("foo")
      expect(described_class.pluralize("foo", 2)).to eq("foos")
    end

    it "combines the stem with the singular suffix based on the count" do
      expect(described_class.pluralize("foo", 0, singular: "o")).to eq("foos")
      expect(described_class.pluralize("foo", 1, singular: "o")).to eq("fooo")
      expect(described_class.pluralize("foo", 2, singular: "o")).to eq("foos")
    end

    it "combines the stem with the plural suffix based on the count" do
      expect(described_class.pluralize("foo", 0, plural: "es")).to eq("fooes")
      expect(described_class.pluralize("foo", 1, plural: "es")).to eq("foo")
      expect(described_class.pluralize("foo", 2, plural: "es")).to eq("fooes")
    end

    it "combines the stem with the singular and plural suffix based on the count" do
      expect(described_class.pluralize("foo", 0, singular: "o", plural: "es")).to eq("fooes")
      expect(described_class.pluralize("foo", 1, singular: "o", plural: "es")).to eq("fooo")
      expect(described_class.pluralize("foo", 2, singular: "o", plural: "es")).to eq("fooes")
    end
  end

  describe ".to_sentence" do
    it "converts a plain array to a sentence" do
      expect(described_class.to_sentence([])).to eq("")
      expect(described_class.to_sentence(["one"])).to eq("one")
      expect(described_class.to_sentence(["one", "two"])).to eq("one and two")
      expect(described_class.to_sentence(["one", "two", "three"])).to eq("one, two, and three")
    end

    it "converts an array to a sentence with a custom connector" do
      expect(described_class.to_sentence(["one", "two", "three"], words_connector: " ")).to eq("one two, and three")
      expect(described_class.to_sentence(["one", "two", "three"],
                                         words_connector: " & ")).to eq("one & two, and three")
    end

    it "converts an array to a sentence with a custom last word connector" do
      expect(described_class.to_sentence(["one", "two", "three"],
                                         last_word_connector: ", and also ")).to eq("one, two, and also three")
      expect(described_class.to_sentence(["one", "two", "three"], last_word_connector: " ")).to eq("one, two three")
      expect(described_class.to_sentence(["one", "two", "three"],
                                         last_word_connector: " and ")).to eq("one, two and three")
    end

    it "converts an array to a sentence with a custom two word connector" do
      expect(described_class.to_sentence(["one", "two"], two_words_connector: " ")).to eq("one two")
    end

    it "creates a new string" do
      elements = ["one"]
      expect(described_class.to_sentence(elements).object_id).not_to eq(elements[0].object_id)
    end

    it "converts a non-String to a sentence" do
      expect(described_class.to_sentence([1])).to eq("1")
    end

    it "converts an array with blank elements to a sentence" do
      expect(described_class.to_sentence([nil, "one", "", "two", "three"])).to eq(", one, , two, and three")
    end

    it "does not return a frozen string" do
      expect(described_class.to_sentence([])).not_to be_frozen
      expect(described_class.to_sentence(["one"])).not_to be_frozen
      expect(described_class.to_sentence(["one", "two"])).not_to be_frozen
      expect(described_class.to_sentence(["one", "two", "three"])).not_to be_frozen
    end
  end
end
