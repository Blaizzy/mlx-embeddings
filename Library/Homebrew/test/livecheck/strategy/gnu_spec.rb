# typed: false
# frozen_string_literal: true

require "livecheck/strategy/gnu"

describe Homebrew::Livecheck::Strategy::Gnu do
  subject(:gnu) { described_class }

  let(:gnu_url) { "https://ftp.gnu.org/gnu/abc/def-1.2.3.tar.gz" }
  let(:savannah_gnu_url) { "https://download.savannah.gnu.org/releases/abc/def-1.2.3.tar.gz" }
  let(:non_gnu_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a non-Savannah GNU URL" do
      expect(gnu.match?(gnu_url)).to be true
    end

    it "returns false if the argument provided is a Savannah GNU URL" do
      expect(gnu.match?(savannah_gnu_url)).to be false
    end

    it "returns false if the argument provided is not a GNU URL" do
      expect(gnu.match?(non_gnu_url)).to be false
    end
  end
end
