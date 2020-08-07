# frozen_string_literal: true

require "formula"
require "livecheck"

describe Livecheck do
  HOMEPAGE_URL = "https://example.com/"
  STABLE_URL = "https://example.com/example-1.2.3.tar.gz"
  HEAD_URL = "https://example.com/example.git"

  let(:f) do
    formula do
      homepage HOMEPAGE_URL
      url STABLE_URL
      head HEAD_URL
    end
  end
  let(:livecheckable) { described_class.new(f) }

  describe "#regex" do
    it "returns nil if not set" do
      expect(livecheckable.regex).to be nil
    end

    it "returns the Regexp if set" do
      livecheckable.regex(/foo/)
      expect(livecheckable.regex).to eq(/foo/)
    end

    it "raises a TypeError if the argument isn't a Regexp" do
      expect {
        livecheckable.regex("foo")
      }.to raise_error(TypeError, "Livecheck#regex expects a Regexp")
    end
  end

  describe "#skip" do
    it "sets @skip to true when no argument is provided" do
      expect(livecheckable.skip).to be true
      expect(livecheckable.instance_variable_get(:@skip)).to be true
      expect(livecheckable.instance_variable_get(:@skip_msg)).to be nil
    end

    it "sets @skip to true and @skip_msg to the provided String" do
      expect(livecheckable.skip("foo")).to be true
      expect(livecheckable.instance_variable_get(:@skip)).to be true
      expect(livecheckable.instance_variable_get(:@skip_msg)).to eq("foo")
    end

    it "raises a TypeError if the argument isn't a String" do
      expect {
        livecheckable.skip(/foo/)
      }.to raise_error(TypeError, "Livecheck#skip expects a String")
    end
  end

  describe "#skip?" do
    it "returns the value of @skip" do
      expect(livecheckable.skip?).to be false

      livecheckable.skip
      expect(livecheckable.skip?).to be true
    end
  end

  describe "#strategy" do
    it "returns nil if not set" do
      expect(livecheckable.strategy).to be nil
    end

    it "returns the Symbol if set" do
      livecheckable.strategy(:page_match)
      expect(livecheckable.strategy).to eq(:page_match)
    end

    it "raises a TypeError if the argument isn't a Symbol" do
      expect {
        livecheckable.strategy("page_match")
      }.to raise_error(TypeError, "Livecheck#strategy expects a Symbol")
    end
  end

  describe "#url" do
    it "returns nil if not set" do
      expect(livecheckable.url).to be nil
    end

    it "returns the URL if set" do
      livecheckable.url("foo")
      expect(livecheckable.url).to eq("foo")

      livecheckable.url(:homepage)
      expect(livecheckable.url).to eq(HOMEPAGE_URL)

      livecheckable.url(:stable)
      expect(livecheckable.url).to eq(STABLE_URL)

      livecheckable.url(:head)
      expect(livecheckable.url).to eq(HEAD_URL)
    end

    it "raises a TypeError if the argument isn't a String or Symbol" do
      expect {
        livecheckable.url(/foo/)
      }.to raise_error(TypeError, "Livecheck#url expects a String or valid Symbol")
    end
  end

  describe "#to_hash" do
    it "returns a Hash of all instance variables" do
      expect(livecheckable.to_hash).to eq(
        {
          "regex"    => nil,
          "skip"     => false,
          "skip_msg" => nil,
          "strategy" => nil,
          "url"      => nil,
        },
      )
    end
  end
end
