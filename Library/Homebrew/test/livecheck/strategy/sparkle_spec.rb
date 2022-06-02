# typed: false
# frozen_string_literal: true

require "livecheck/strategy"
require "bundle_version"

describe Homebrew::Livecheck::Strategy::Sparkle do
  subject(:sparkle) { described_class }

  let(:appcast_url) { "https://www.example.com/example/appcast.xml" }
  let(:non_http_url) { "ftp://brew.sh/" }

  let(:item_hash) {
    [
      {
        title:         "Version 1.2.3",
        pub_date:      "Fri, 01 Jan 2021 01:23:45 +0000",
        url:           "https://www.example.com/example/example-1.2.3.tar.gz",
        short_version: "1.2.3",
        version:       "123",
      },
      {
        title:         "Version 1.2.2",
        pub_date:      "Not a parseable date string",
        url:           "https://www.example.com/example/example-1.2.2.tar.gz",
        short_version: "1.2.2",
        version:       "122",
      },
    ]
  }

  let(:xml) {
    first_item = <<~EOS
      <item>
        <title>#{item_hash[0][:title]}</title>
        <sparkle:minimumSystemVersion>10.10</sparkle:minimumSystemVersion>
        <sparkle:releaseNotesLink>https://www.example.com/example/#{item_hash[0][:short_version]}.html</sparkle:releaseNotesLink>
        <pubDate>#{item_hash[0][:pub_date]}</pubDate>
        <enclosure url="#{item_hash[0][:url]}" sparkle:shortVersionString="#{item_hash[0][:short_version]}" sparkle:version="#{item_hash[0][:version]}" length="12345678" type="application/octet-stream" sparkle:dsaSignature="ABCDEF+GHIJKLMNOPQRSTUVWXYZab/cdefghijklmnopqrst/uvwxyz1234567==" />
      </item>
    EOS

    second_item = <<~EOS
      <item>
        <title>#{item_hash[1][:title]}</title>
        <sparkle:minimumSystemVersion>10.10</sparkle:minimumSystemVersion>
        <sparkle:releaseNotesLink>https://www.example.com/example/#{item_hash[1][:short_version]}.html</sparkle:releaseNotesLink>
        <pubDate>#{item_hash[1][:pub_date]}</pubDate>
        <sparkle:version>#{item_hash[1][:version]}</sparkle:version>
        <sparkle:shortVersionString>#{item_hash[1][:short_version]}</sparkle:shortVersionString>
        <link>#{item_hash[1][:url]}</link>
      </item>
    EOS

    items_to_omit = <<~EOS
      #{first_item.sub(%r{<(enclosure[^>]+?)\s*?/>}, '<\1 os="not-osx" />')}
      #{first_item.sub(/(<sparkle:minimumSystemVersion>)[^<]+?</m, '\1100<')}
      #{first_item.sub(/(<sparkle:minimumSystemVersion>)[^<]+?</m, '\19000<')}
      <item>
      </item>
    EOS

    appcast = <<~EOS
      <?xml version="1.0" encoding="utf-8"?>
      <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
        <channel>
          <title>Example Changelog</title>
          <link>#{appcast_url}</link>
          <description>Most recent changes with links to updates.</description>
          <language>en</language>
          #{first_item}
          #{second_item}
        </channel>
      </rss>
    EOS

    omitted_items = appcast.sub("</item>", "</item>\n#{items_to_omit}")
    beta_channel_item = appcast.sub(
      first_item,
      first_item.sub(
        "</title",
        "</title>\n<sparkle:channel>beta</sparkle:channel>",
      ),
    )
    no_versions_item =
      appcast
        .sub(second_item, "")
        .gsub(/sparkle:(shortVersionString|version)="[^"]+?"\s*/, "")
        .sub(
          "<title>#{item_hash[0][:title]}</title>",
          "<title>Version</title>",
        )
    no_items = appcast.sub(%r{<item>.+</item>}m, "")
    undefined_namespace = appcast.sub(/\s*xmlns:sparkle="[^"]+?"/, "")

    {
      appcast:             appcast,
      omitted_items:       omitted_items,
      beta_channel_item:   beta_channel_item,
      no_versions_item:    no_versions_item,
      no_items:            no_items,
      undefined_namespace: undefined_namespace,
    }
  }

  let(:title_regex) { /Version\s+v?(\d+(?:\.\d+)+)\s*$/i }

  let(:items) {
    items = {
      appcast: [
        Homebrew::Livecheck::Strategy::Sparkle::Item.new(
          title:          item_hash[0][:title],
          pub_date:       Time.parse(item_hash[0][:pub_date]),
          url:            item_hash[0][:url],
          bundle_version: Homebrew::BundleVersion.new(item_hash[0][:short_version], item_hash[0][:version]),
        ),
        Homebrew::Livecheck::Strategy::Sparkle::Item.new(
          title:          item_hash[1][:title],
          pub_date:       Time.new(0),
          url:            item_hash[1][:url],
          bundle_version: Homebrew::BundleVersion.new(item_hash[1][:short_version], item_hash[1][:version]),
        ),
      ],
    }

    beta_channel_item = items[:appcast][0].clone
    beta_channel_item.channel = "beta"
    items[:beta_channel_item] = [beta_channel_item, items[:appcast][1].clone]

    no_versions_item = items[:appcast][0].clone
    no_versions_item.title = "Version"
    no_versions_item.bundle_version = nil
    items[:no_versions_item] = [no_versions_item]

    items
  }

  let(:versions) { [items[:appcast][0].nice_version] }

  describe "::match?" do
    it "returns true for an HTTP URL" do
      expect(sparkle.match?(appcast_url)).to be true
    end

    it "returns false for a non-HTTP URL" do
      expect(sparkle.match?(non_http_url)).to be false
    end
  end

  describe "::items_from_content" do
    let(:items_from_appcast) { sparkle.items_from_content(xml[:appcast]) }
    let(:first_item) { items_from_appcast[0] }

    it "returns nil if content is blank" do
      expect(sparkle.items_from_content("")).to eq([])
    end

    it "returns an array of Items when given XML data" do
      expect(items_from_appcast).to eq(items[:appcast])
      expect(first_item.title).to eq(item_hash[0][:title])
      expect(first_item.pub_date).to eq(Time.parse(item_hash[0][:pub_date]))
      expect(first_item.url).to eq(item_hash[0][:url])
      expect(first_item.short_version).to eq(item_hash[0][:short_version])
      expect(first_item.version).to eq(item_hash[0][:version])

      expect(sparkle.items_from_content(xml[:beta_channel_item])).to eq(items[:beta_channel_item])
      expect(sparkle.items_from_content(xml[:no_versions_item])).to eq(items[:no_versions_item])
    end
  end

  describe "::versions_from_content" do
    let(:subbed_items) { items[:appcast].map { |item| item.nice_version.sub("1", "0") } }

    it "returns an array of version strings when given content" do
      expect(sparkle.versions_from_content(xml[:appcast])).to eq(versions)
      expect(sparkle.versions_from_content(xml[:omitted_items])).to eq(versions)
      expect(sparkle.versions_from_content(xml[:beta_channel_item])).to eq(versions)
      expect(sparkle.versions_from_content(xml[:no_versions_item])).to eq([])
      expect(sparkle.versions_from_content(xml[:undefined_namespace])).to eq(versions)
    end

    it "returns an empty array if no items are found" do
      expect(sparkle.versions_from_content(xml[:no_items])).to eq([])
    end

    it "returns an array of version strings when given content and a block" do
      # Returning a string from block
      expect(
        sparkle.versions_from_content(xml[:appcast]) do |item|
          item.nice_version&.sub("1", "0")
        end,
      ).to eq([subbed_items[0]])

      # Returning an array of strings from block
      expect(
        sparkle.versions_from_content(xml[:appcast]) do |items|
          items.map { |item| item.nice_version&.sub("1", "0") }
        end,
      ).to eq(subbed_items)

      expect(
        sparkle.versions_from_content(xml[:beta_channel_item]) do |items|
          items.find { |item| item.channel.nil? }&.nice_version
        end,
      ).to eq([items[:appcast][1].nice_version])
    end

    it "returns an array of version strings when given content, a regex, and a block" do
      # Returning a string from the block
      expect(
        sparkle.versions_from_content(xml[:appcast], title_regex) do |item, regex|
          item.title[regex, 1]
        end,
      ).to eq([item_hash[0][:short_version]])

      expect(
        sparkle.versions_from_content(xml[:appcast], title_regex) do |items, regex|
          next if (item = items[0]).blank?

          match = item&.title&.match(regex)
          next if match.blank?

          "#{match[1]},#{item.version}"
        end,
      ).to eq(["#{item_hash[0][:short_version]},#{item_hash[0][:version]}"])

      # Returning an array of strings from the block
      expect(
        sparkle.versions_from_content(xml[:appcast], title_regex) do |item, regex|
          [item.title[regex, 1]]
        end,
      ).to eq([item_hash[0][:short_version]])

      expect(
        sparkle.versions_from_content(xml[:appcast], &:short_version),
      ).to eq([item_hash[0][:short_version]])

      expect(
        sparkle.versions_from_content(xml[:appcast], title_regex) do |items, regex|
          items.map { |item| item.title[regex, 1] }
        end,
      ).to eq(items[:appcast].map(&:short_version))
    end

    it "allows a nil return from a block" do
      expect(
        sparkle.versions_from_content(xml[:appcast]) do |item|
          _ = item # To appease `brew style` without modifying arg name
          next
        end,
      ).to eq([])
    end

    it "errors on an invalid return type from a block" do
      expect {
        sparkle.versions_from_content(xml[:appcast]) do |item|
          _ = item # To appease `brew style` without modifying arg name
          123
        end
      }.to raise_error(TypeError, Homebrew::Livecheck::Strategy::INVALID_BLOCK_RETURN_VALUE_MSG)
    end

    it "errors if the first block argument uses an unhandled name" do
      expect { sparkle.versions_from_content(xml[:appcast]) { |something| something } }
        .to raise_error("First argument of Sparkle `strategy` block must be `item` or `items`")
    end
  end
end
