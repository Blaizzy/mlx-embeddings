# typed: false
# frozen_string_literal: true

require "livecheck/strategy/cpan"

describe Homebrew::Livecheck::Strategy::Cpan do
  subject(:cpan) { described_class }

  let(:cpan_url_no_subdirectory) { "https://cpan.metacpan.org/authors/id/H/HO/HOMEBREW/Brew-v1.2.3.tar.gz" }
  let(:cpan_url_with_subdirectory) { "https://cpan.metacpan.org/authors/id/H/HO/HOMEBREW/brew/brew-v1.2.3.tar.gz" }
  let(:non_cpan_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a CPAN URL" do
      expect(cpan.match?(cpan_url_no_subdirectory)).to be true
      expect(cpan.match?(cpan_url_with_subdirectory)).to be true
    end

    it "returns false if the argument provided is not a CPAN URL" do
      expect(cpan.match?(non_cpan_url)).to be false
    end
  end
end
