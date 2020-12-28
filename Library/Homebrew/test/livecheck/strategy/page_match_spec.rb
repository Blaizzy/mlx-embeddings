# typed: false
# frozen_string_literal: true

require "livecheck/strategy/page_match"

describe Homebrew::Livecheck::Strategy::PageMatch do
  subject(:page_match) { described_class }

  let(:url) { "https://brew.sh/blog/" }
  let(:regex) { %r{href=.*?/homebrew[._-]v?(\d+(?:\.\d+)+)/?["' >]}i }

  let(:page_content) {
    <<~EOS
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <title>Homebrew — Homebrew</title>
        </head>
        <body>
          <ul class="posts">
            <li><a href="/2020/12/01/homebrew-2.6.0/" title="2.6.0"><h2>2.6.0</h2><h3>01 Dec 2020</h3></a></li>
            <li><a href="/2020/11/18/homebrew-tap-with-bottles-uploaded-to-github-releases/" title="Homebrew tap with bottles uploaded to GitHub Releases"><h2>Homebrew tap with bottles uploaded to GitHub Releases</h2><h3>18 Nov 2020</h3></a></li>
            <li><a href="/2020/09/08/homebrew-2.5.0/" title="2.5.0"><h2>2.5.0</h2><h3>08 Sep 2020</h3></a></li>
            <li><a href="/2020/06/11/homebrew-2.4.0/" title="2.4.0"><h2>2.4.0</h2><h3>11 Jun 2020</h3></a></li>
            <li><a href="/2020/05/29/homebrew-2.3.0/" title="2.3.0"><h2>2.3.0</h2><h3>29 May 2020</h3></a></li>
            <li><a href="/2019/11/27/homebrew-2.2.0/" title="2.2.0"><h2>2.2.0</h2><h3>27 Nov 2019</h3></a></li>
            <li><a href="/2019/06/14/homebrew-maintainer-meeting/" title="Homebrew Maintainer Meeting"><h2>Homebrew Maintainer Meeting</h2><h3>14 Jun 2019</h3></a></li>
            <li><a href="/2019/04/04/homebrew-2.1.0/" title="2.1.0"><h2>2.1.0</h2><h3>04 Apr 2019</h3></a></li>
            <li><a href="/2019/02/02/homebrew-2.0.0/" title="2.0.0"><h2>2.0.0</h2><h3>02 Feb 2019</h3></a></li>
            <li><a href="/2019/01/09/homebrew-1.9.0/" title="1.9.0"><h2>1.9.0</h2><h3>09 Jan 2019</h3></a></li>
          </ul>
        </body>
      </html>
    EOS
  }

  let(:page_content_matches) { ["2.6.0", "2.5.0", "2.4.0", "2.3.0", "2.2.0", "2.1.0", "2.0.0", "1.9.0"] }

  let(:find_versions_return_hash) {
    {
      matches: {
        "2.6.0" => Version.new("2.6.0"),
        "2.5.0" => Version.new("2.5.0"),
        "2.4.0" => Version.new("2.4.0"),
        "2.3.0" => Version.new("2.3.0"),
        "2.2.0" => Version.new("2.2.0"),
        "2.1.0" => Version.new("2.1.0"),
        "2.0.0" => Version.new("2.0.0"),
        "1.9.0" => Version.new("1.9.0"),
      },
      regex:   regex,
      url:     url,
    }
  }

  let(:find_versions_cached_return_hash) {
    return_hash = find_versions_return_hash
    return_hash[:cached] = true
    return_hash
  }

  describe "::match?" do
    it "returns true for any URL" do
      expect(page_match.match?(url)).to be true
    end
  end

  describe "::page_matches" do
    it "finds matching text in page content using a regex" do
      expect(page_match.page_matches(page_content, regex)).to eq(page_content_matches)
    end

    it "finds matching text in page content using a strategy block" do
      expect(page_match.page_matches(page_content, regex) { |content| content.scan(regex).map(&:first).uniq })
        .to eq(page_content_matches)
    end
  end

  describe "::find_versions?" do
    it "finds versions in provided_content" do
      expect(page_match.find_versions(url, regex, page_content)).to eq(find_versions_cached_return_hash)
    end
  end
end
