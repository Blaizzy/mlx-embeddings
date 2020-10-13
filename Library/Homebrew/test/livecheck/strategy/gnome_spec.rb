# typed: false
# frozen_string_literal: true

require "livecheck/strategy/gnome"

describe Homebrew::Livecheck::Strategy::Gnome do
  subject(:gnome) { described_class }

  let(:gnome_url) { "https://download.gnome.org/sources/abc/1.2/def-1.2.3.tar.xz" }
  let(:non_gnome_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a GNOME URL" do
      expect(gnome.match?(gnome_url)).to be true
    end

    it "returns false if the argument provided is not a GNOME URL" do
      expect(gnome.match?(non_gnome_url)).to be false
    end
  end
end
