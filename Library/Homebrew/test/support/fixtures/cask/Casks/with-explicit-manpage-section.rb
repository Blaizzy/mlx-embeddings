cask 'with-explicit-manpage-section' do
  version '1.2.3'
  sha256 '67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94'

  url "file://#{TEST_FIXTURE_DIR}/cask/AppWithManpage.zip"
  homepage 'https://brew.sh/with-explicit-manpage-section'

  manpage 'manpage.1', section: 3
end
