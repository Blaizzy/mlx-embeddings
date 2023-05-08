# typed: true
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {GithubLatest} strategy identifies versions of software at
      # github.com by checking a repository's "latest" release using the
      # GitHub API.
      #
      # GitHub URLs take a few different formats:
      #
      # * `https://github.com/example/example/releases/download/1.2.3/example-1.2.3.tar.gz`
      # * `https://github.com/example/example/archive/v1.2.3.tar.gz`
      # * `https://github.com/downloads/example/example/example-1.2.3.tar.gz`
      #
      # {GithubLatest} should only be used when the upstream repository has a
      # "latest" release for a suitable version and the strategy is necessary
      # or appropriate (e.g. {Git} returns an unreleased version or the
      # `stable` URL is a release asset). The strategy can only be applied by
      # using `strategy :github_latest` in a `livecheck` block.
      #
      # The default regex identifies versions like `1.2.3`/`v1.2.3` in the
      # release's tag name. This is a common tag format but a modified regex
      # can be provided in a `livecheck` block to override the default if a
      # repository uses a different format (e.g. `1.2.3d`, `1.2.3-4`, etc.).
      #
      # @api public
      class GithubLatest
        NICE_NAME = "GitHub - Latest"

        # A priority of zero causes livecheck to skip the strategy. We do this
        # for {GithubLatest} so we can selectively apply the strategy using
        # `strategy :github_latest` in a `livecheck` block.
        PRIORITY = 0

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{
          ^https?://github\.com
          /(?:downloads/)?(?<username>[^/]+) # The GitHub username
          /(?<repository>[^/]+)              # The GitHub repository name
        }ix.freeze

        # The default regex used to identify a version from a tag when a regex
        # isn't provided.
        DEFAULT_REGEX = /v?(\d+(?:\.\d+)+)/i.freeze

        # Keys in the release JSON that could contain the version.
        # Tag name first since that is closer to other livechecks.
        VERSION_KEYS = ["tag_name", "name"].freeze

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          URL_MATCH_REGEX.match?(url)
        end

        # Extracts information from a provided URL and uses it to generate
        # various input values used by the strategy to check for new versions.
        # Some of these values act as defaults and can be overridden in a
        # `livecheck` block.
        #
        # @param url [String] the URL used to generate values
        # @return [Hash]
        sig { params(url: String).returns(T::Hash[Symbol, T.untyped]) }
        def self.generate_input_values(url)
          values = {}

          match = url.sub(/\.git$/i, "").match(URL_MATCH_REGEX)
          return values if match.blank?

          values[:url] = "https://api.github.com/repos/#{match[:username]}/#{match[:repository]}/releases/latest"
          values[:username] = match[:username]
          values[:repository] = match[:repository]

          values
        end

        # Uses a regex to match the version from release JSON or, if a block is
        # provided, passes the JSON to the block to handle matching. With
        # either approach, an array of unique matches is returned.
        #
        # @param content [Array, Hash] list of releases or a single release
        # @param regex [Regexp] a regex used for matching versions in the content
        # @param block [Proc, nil] a block to match the content
        # @return [Array]
        sig {
          params(
            content: T.any(T::Array[T::Hash[String, T.untyped]], T::Hash[String, T.untyped]),
            regex:   Regexp,
            block:   T.nilable(Proc),
          ).returns(T::Array[String])
        }
        def self.versions_from_content(content, regex, &block)
          if block.present?
            block_return_value = if regex.present?
              yield(content, regex)
            else
              yield(content)
            end
            return Strategy.handle_block_return(block_return_value)
          end

          content = [content] unless content.is_a?(Array)
          content.reject(&:blank?).map do |release|
            next if release["draft"] || release["prerelease"]

            value = T.let(nil, T.untyped)
            VERSION_KEYS.find do |key|
              match = release[key]&.match(regex)
              next if match.blank?

              value = match[1]
            end
            value
          end.compact.uniq
        end

        # Generates the GitHub API URL for the repository's "latest" release
        # and identifies the version from the JSON response.
        #
        # @param url [String] the URL of the content to check
        # @param regex [Regexp] a regex used for matching versions in content
        # @return [Hash]
        sig {
          params(
            url:     String,
            regex:   Regexp,
            _unused: T.nilable(T::Hash[Symbol, T.untyped]),
            block:   T.nilable(Proc),
          ).returns(T::Hash[Symbol, T.untyped])
        }
        def self.find_versions(url:, regex: DEFAULT_REGEX, **_unused, &block)
          match_data = { matches: {}, regex: regex, url: url }

          generated = generate_input_values(url)
          return match_data if generated.blank?

          match_data[:url] = generated[:url]

          release = GitHub.get_latest_release(generated[:username], generated[:repository])
          versions_from_content(release, regex, &block).each do |match_text|
            match_data[:matches][match_text] = Version.new(match_text)
          end

          match_data
        end
      end
    end
    GitHubLatest = Homebrew::Livecheck::Strategy::GithubLatest
  end
end
