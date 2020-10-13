# typed: false
# frozen_string_literal: true

require "livecheck/strategy"

describe Homebrew::Livecheck::Strategy do
  subject(:strategy) { described_class }

  describe "::from_symbol" do
    it "returns the Strategy module represented by the Symbol argument" do
      expect(strategy.from_symbol(:page_match)).to eq(Homebrew::Livecheck::Strategy::PageMatch)
    end
  end

  describe "::from_url" do
    let(:url) { "https://sourceforge.net/projects/test" }

    context "when no regex is provided" do
      it "returns an array of usable strategies which doesn't include PageMatch" do
        expect(strategy.from_url(url)).to eq([Homebrew::Livecheck::Strategy::Sourceforge])
      end
    end

    context "when a regex is provided" do
      it "returns an array of usable strategies including PageMatch, sorted in descending order by priority" do
        expect(strategy.from_url(url, regex_provided: true))
          .to eq(
            [Homebrew::Livecheck::Strategy::Sourceforge, Homebrew::Livecheck::Strategy::PageMatch],
          )
      end
    end
  end
end
