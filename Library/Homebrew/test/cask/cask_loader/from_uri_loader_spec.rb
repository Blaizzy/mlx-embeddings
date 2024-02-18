# frozen_string_literal: true

RSpec.describe Cask::CaskLoader::FromURILoader do
  describe "::try_new" do
    it "returns a loader when given an URI" do
      expect(described_class.try_new(URI("https://brew.sh/"))).not_to be_nil
    end

    it "returns a loader when given a string which can be parsed to a URI" do
      expect(described_class.try_new("https://brew.sh/")).not_to be_nil
    end

    it "returns nil when given a string with Cask contents containing a URL" do
      expect(described_class.try_new(<<~RUBY)).to be_nil
        cask 'token' do
          url 'https://brew.sh/'
        end
      RUBY
    end
  end
end
