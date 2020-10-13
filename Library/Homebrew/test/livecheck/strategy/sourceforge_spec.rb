# typed: false
# frozen_string_literal: true

require "livecheck/strategy/sourceforge"

describe Homebrew::Livecheck::Strategy::Sourceforge do
  subject(:sourceforge) { described_class }

  let(:sourceforge_url) { "https://downloads.sourceforge.net/project/abc/def-1.2.3.tar.gz" }
  let(:non_sourceforge_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a SourceForge URL" do
      expect(sourceforge.match?(sourceforge_url)).to be true
    end

    it "returns false if the argument provided is not a SourceForge URL" do
      expect(sourceforge.match?(non_sourceforge_url)).to be false
    end
  end
end
