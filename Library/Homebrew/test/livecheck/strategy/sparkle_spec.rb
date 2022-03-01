# typed: false
# frozen_string_literal: true

require "livecheck/strategy"
require "bundle_version"

describe Homebrew::Livecheck::Strategy::Sparkle do
  subject(:sparkle) { described_class }

  let(:appcast_url) { "https://www.example.com/example/appcast.xml" }
  let(:non_http_url) { "ftp://brew.sh/" }

  let(:appcast_data) {
    {
      title:         "Version 1.2.3",
      pub_date:      "Fri, 01 Jan 2021 01:23:45 +0000",
      url:           "https://www.example.com/example/example.tar.gz",
      short_version: "1.2.3",
      version:       "1234",
    }
  }

  let(:appcast_xml) {
    <<~EOS
      <?xml version="1.0" encoding="utf-8"?>
      <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
        <channel>
          <title>Example Changelog</title>
          <link>#{appcast_url}</link>
          <description>Most recent changes with links to updates.</description>
          <language>en</language>
          <item>
            <title>#{appcast_data[:title]}</title>
            <sparkle:minimumSystemVersion>10.10</sparkle:minimumSystemVersion>
            <sparkle:releaseNotesLink>https://www.example.com/example/1.2.3.html</sparkle:releaseNotesLink>
            <pubDate>#{appcast_data[:pub_date]}</pubDate>
            <enclosure url="#{appcast_data[:url]}" sparkle:shortVersionString="#{appcast_data[:short_version]}" sparkle:version="#{appcast_data[:version]}" length="12345678" type="application/octet-stream" sparkle:dsaSignature="ABCDEF+GHIJKLMNOPQRSTUVWXYZab/cdefghijklmnopqrst/uvwxyz1234567==" />
          </item>
        </channel>
      </rss>
    EOS
  }

  let(:title_regex) { /Version\s+v?(\d+(?:\.\d+)+)\s*$/i }

  let(:item) {
    Homebrew::Livecheck::Strategy::Sparkle::Item.new(
      title:          appcast_data[:title],
      pub_date:       Time.parse(appcast_data[:pub_date]),
      url:            appcast_data[:url],
      bundle_version: Homebrew::BundleVersion.new(appcast_data[:short_version], appcast_data[:version]),
    )
  }

  let(:versions) { [item.bundle_version.nice_version] }

  describe "::match?" do
    it "returns true for an HTTP URL" do
      expect(sparkle.match?(appcast_url)).to be true
    end

    it "returns false for a non-HTTP URL" do
      expect(sparkle.match?(non_http_url)).to be false
    end
  end

  describe "::item_from_content" do
    let(:item_from_appcast_xml) { sparkle.item_from_content(appcast_xml) }

    it "returns nil if content is blank" do
      expect(sparkle.item_from_content("")).to be_nil
    end

    it "returns an Item when given XML data" do
      expect(item_from_appcast_xml).to be_a(Homebrew::Livecheck::Strategy::Sparkle::Item)
      expect(item_from_appcast_xml).to eq(item)
      expect(item_from_appcast_xml.title).to eq(appcast_data[:title])
      expect(item_from_appcast_xml.pub_date).to eq(Time.parse(appcast_data[:pub_date]))
      expect(item_from_appcast_xml.url).to eq(appcast_data[:url])
      expect(item_from_appcast_xml.short_version).to eq(appcast_data[:short_version])
      expect(item_from_appcast_xml.version).to eq(appcast_data[:version])
    end
  end

  describe "::versions_from_content" do
    it "returns an array of version strings when given content" do
      expect(sparkle.versions_from_content(appcast_xml)).to eq(versions)
    end

    it "returns an array of version strings when given content and a block" do
      # Returning a string from block
      expect(
        sparkle.versions_from_content(appcast_xml) do |item|
          item.bundle_version&.nice_version&.sub("3", "4")
        end,
      ).to eq([item.bundle_version.nice_version.sub("3", "4")])

      # Returning an array of strings from block (unlikely to be used)
      expect(sparkle.versions_from_content(appcast_xml) { versions }).to eq(versions)
    end

    it "returns an array of version strings when given content, a regex, and a block" do
      # Returning a string from block
      expect(
        sparkle.versions_from_content(appcast_xml, title_regex) do |item, regex|
          item.title[regex, 1]
        end,
      ).to eq([item.bundle_version.short_version])

      # Returning an array of strings from block (unlikely to be used)
      expect(
        sparkle.versions_from_content(appcast_xml, title_regex) do |item, regex|
          [item.title[regex, 1]]
        end,
      ).to eq([item.bundle_version.short_version])
    end

    it "allows a nil return from a block" do
      expect(sparkle.versions_from_content(appcast_xml) { next }).to eq([])
    end

    it "errors on an invalid return type from a block" do
      expect { sparkle.versions_from_content(appcast_xml) { 123 } }
        .to raise_error(TypeError, Homebrew::Livecheck::Strategy::INVALID_BLOCK_RETURN_VALUE_MSG)
    end
  end
end
