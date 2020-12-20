# typed: false
# frozen_string_literal: true

require "livecheck/strategy/sparkle"

describe Homebrew::Livecheck::Strategy::Sparkle do
  subject(:sparkle) { described_class }

  let(:url) { "https://www.example.com/example/appcast.xml" }

  let(:appcast_data) {
    {
      title:          "Version 1.2.3",
      url:            "https://www.example.com/example/example.tar.gz",
      bundle_version: Homebrew::BundleVersion.new("1.2.3", "1234"),
      short_version:  "1.2.3",
      version:        "1234",
    }
  }

  let(:appcast_item) {
    Homebrew::Livecheck::Strategy::Sparkle::Item.new(
      {
        title:          appcast_data[:title],
        url:            appcast_data[:url],
        bundle_version: appcast_data[:bundle_version],
        short_version:  appcast_data[:bundle_version]&.short_version,
        version:        appcast_data[:bundle_version]&.version,
      },
    )
  }

  let(:appcast_xml) {
    <<~EOS
      <?xml version="1.0" encoding="utf-8"?>
      <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
        <channel>
          <title>Example Changelog</title>
          <link>#{url}</link>
          <description>Most recent changes with links to updates.</description>
          <language>en</language>
          <item>
            <title>#{appcast_data[:title]}</title>
            <sparkle:minimumSystemVersion>10.10</sparkle:minimumSystemVersion>
            <sparkle:releaseNotesLink>https://www.example.com/example/1.2.3.html</sparkle:releaseNotesLink>
            <enclosure url="#{appcast_data[:url]}" sparkle:shortVersionString="#{appcast_data[:short_version]}" sparkle:version="#{appcast_data[:version]}" length="12345678" type="application/octet-stream" sparkle:dsaSignature="ABCDEF+GHIJKLMNOPQRSTUVWXYZab/cdefghijklmnopqrst/uvwxyz1234567==" />
          </item>
        </channel>
      </rss>
    EOS
  }

  describe "::match?" do
    it "returns true for any URL" do
      expect(sparkle.match?(url)).to be true
    end
  end

  describe "::item_from_content" do
    let(:item_from_appcast_xml) { sparkle.item_from_content(appcast_xml) }

    it "returns nil if content is blank" do
      expect(sparkle.item_from_content("")).to be nil
    end

    it "returns an Item when given XML data" do
      expect(item_from_appcast_xml).to be_a(Homebrew::Livecheck::Strategy::Sparkle::Item)
      expect(item_from_appcast_xml.title).to eq(appcast_item.title)
      expect(item_from_appcast_xml.url).to eq(appcast_item.url)
      expect(item_from_appcast_xml.bundle_version.short_version).to eq(appcast_item.bundle_version.short_version)
      expect(item_from_appcast_xml.bundle_version.version).to eq(appcast_item.bundle_version.version)
    end
  end
end
